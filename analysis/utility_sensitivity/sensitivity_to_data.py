import os
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

from analysis.supporting_analyses.level_difficulty_and_afr import get_all_metrics
from data import LLM_DISPLAY_NAMES

# custom_colors = ["#99cb9b", "#9898ff", "#fe989a"]
custom_colors = ["#7fa983", "#7878e5", "#e06f71"]  # darker version of the ones above
sns.set_palette(custom_colors + sns.color_palette("colorblind6")[3:])

# sns.set_palette(sns.color_palette("colorblind6"))

COMPUTER_MODERN_FONT = True

if COMPUTER_MODERN_FONT:
    rcParams['font.family'] = 'cmr10'
    rcParams['axes.formatter.use_mathtext'] = True


def scr_vs_afr(fp_dataset: Literal["BoarderlineUser", "BasicUser", "both"]):
    """
    Attacker Failure Rate vs. True Negative Rate
    """

    if fp_dataset == "both":
        fpr_datasets = ["BorderlineUser", "BasicUser"]
    else:
        fpr_datasets = [fp_dataset]

    dfs = list()
    for fprd in fpr_datasets:
        df = get_all_metrics()
        df = df[df["setup"] == "general"]
        df = df.set_index(["llm", "defense"])

        if os.path.isfile(f"tables/benign_long_{fprd}.csv"):
            df2 = pd.read_csv(f"tables/benign_long_{fprd}.csv")
            df["FPR"] = df2.set_index(["llm", "defense"])["hard_FP"] / 100.
        else:
            raise ValueError(f"File tables/benign_long_{fprd}.csv not found")

        df.reset_index(inplace=True)
        df = df[df["defense"].str.startswith("C")]
        df["llm"] = df["llm"].map(LLM_DISPLAY_NAMES)
        df = df[["llm", "defense", "success_rate_session", "FPR"]]

        df["afr"] = 1 - df["success_rate_session"]
        df["tnr"] = 1 - df["FPR"]

        df["dataset"] = fprd

        dfs.append(df)

    df = pd.concat(dfs)

    f = 0.7
    fig, ax = plt.subplots(figsize=(f * 7, f * 4), constrained_layout=True)
    if COMPUTER_MODERN_FONT:
        ax.ticklabel_format(useMathText=True)

    sns.scatterplot(data=df, x="afr", y="tnr", hue="defense", style="llm",
                    style_order=["GPT-3.5-turbo", "GPT-4o-mini", "GPT-4"],
                    size="dataset", sizes=(150, 75), ax=ax
                    )

    if fp_dataset == "both":
        # add vertical lines between min and max of defense/llm
        for llm in df["llm"].unique():
            for d, defense in enumerate(sorted(df["defense"].unique())):
                afrs = df[(df["llm"] == llm) & (df["defense"] == defense)]["afr"]
                tnrs = df[(df["llm"] == llm) & (df["defense"] == defense)]["tnr"]
                ax.vlines(afrs.values[0], tnrs.min(), tnrs.max(), colors=f'C{d}', zorder=-1)

    plt.xlabel("Security (AFR)")
    plt.ylabel("Utility (SCR)")

    handles, labels = plt.gca().get_legend_handles_labels()

    handles = handles[:4] + handles[7:] + [handles[4], handles[6], handles[5]]
    labels = labels[:4] + labels[7:] + [labels[4], labels[6], labels[5]]

    for handle in handles[:-3]:
        handle.set_markersize(10)

    label_dict = {
        "C1": "Substrings (C1)",
        "C2": "LLM check (C2)",
        "C3": "Strong prompt (C3)",
        "D": "Combined (D)",
        "defense": "Level",
        "llm": "LLM",
        "dataset": r"Dataset",
        "ultrachat": "BasicUser",
        "handcrafted_fp": "BorderlineUser"
    }

    labels = [label_dict[l] if l in label_dict else l for l in labels]

    legend = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.xlim(right=1.0)
    plt.savefig(f"plots/scr-afr-scatter_{fp_dataset}.pdf")
    plt.show()


def utility_vs_lambda(fp_dataset: Literal["BorderlineUser", "BasicUser", "both"]):
    if fp_dataset == "both":
        fpr_datasets = ["BorderlineUser", "BasicUser"]
    else:
        fpr_datasets = [fp_dataset]

    dfs = list()
    for fprd in fpr_datasets:
        df = get_all_metrics()
        df = df[df["setup"] == "general"]
        df = df.set_index(["llm", "defense"])

        if os.path.isfile(f"tables/benign_long_{fprd}.csv"):
            df2 = pd.read_csv(f"tables/benign_long_{fprd}.csv")
            df["FPR"] = df2.set_index(["llm", "defense"])["hard_FP"] / 100.
        else:
            raise ValueError(f"File tables/benign_long_{fprd}.csv not found")

        df.reset_index(inplace=True)
        df = df[df["defense"].str.startswith("C")]
        df["llm"] = df["llm"].map(LLM_DISPLAY_NAMES)
        df = df[["llm", "defense", "success_rate_session", "FPR"]]

        df["afr"] = 1 - df["success_rate_session"]
        df["tnr"] = 1 - df["FPR"]

        df["dataset"] = fprd

        dfs.append(df)

    df = pd.concat(dfs)

    dfs = list()
    for lambda_ in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        df2 = df.copy()
        df2["utility"] = (1 - lambda_) * df2["afr"] + lambda_ * df2["tnr"]
        df2["lambda"] = lambda_
        dfs.append(df2)

    df = pd.concat(dfs)

    fig, ax = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
    for i, ds in enumerate(["BasicUser", "BorderlineUser"]):
        sns.lineplot(df[df.dataset == ds],
                     x="lambda", y="utility",
                     hue="defense", style="llm", ax=ax[i],
                     errorbar=None)
        ax[i].set_title(ds)
    plt.savefig("plots/utility_vs_lambda.pdf")
    plt.savefig("plots/utility_vs_lambda.png")
    plt.show()


def main():
    scr_vs_afr("both")


if __name__ == '__main__':
    main()
