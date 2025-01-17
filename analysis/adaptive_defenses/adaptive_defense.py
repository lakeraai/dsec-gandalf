"""
Let's simulate the following scenario.
A user has a certain budget of "suspicious trials", meaning that the session ends
if the defense is triggered x times.

To simulate that, we use the full session Gandalf prompts and resubmit then.

Plot: Block threshold vs session success rate.

Compare defenses B, C1, C2, C3 and (C1+C2+C3)
"""

import os
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from scipy.stats import binom
from tqdm import tqdm

from data import LLM_DISPLAY_NAMES, load_adaptive_defense_experiment

pal = ["#7fa983", "#626ffa", "#f1553a", "#3b3b3b"]
sns.set_palette(pal)

COMPUTER_MODERN_FONT = True

if COMPUTER_MODERN_FONT:
    rcParams["font.family"] = "cmr10"
    rcParams["axes.formatter.use_mathtext"] = True


def session_end_probability(X, p, T):
    """
    Compute the probability that a session ends within its X rounds without using a for loop.

    Parameters:
    X (int): The total number of rounds.
    p (float): The probability that any given round is flagged.
    T (int): The threshold for ending the session.

    Returns:
    float: The probability that the session ends on or before round X.
    """
    return 1 - binom.cdf(T - 1, X, p)


# calculate hypothetical session success rates
def get_ssr(df: pd.DataFrame, defense: str, threshold: int):
    success = df["success"] & (df[f"n_blocks_{defense}"] < threshold)
    ssr = success.sum() / len(success)
    return ssr


def get_fpr(fp_dataset: str, df: pd.DataFrame, defense: str, threshold: int):
    if not os.path.isfile(f"tables/benign_long_{fp_dataset}.csv"):
        raise FileNotFoundError(
            "Could not find file."
            "Please run analysis/supporting_analyses/false_positives.py first."
            "Also check path from which this script is executed."
        )

    fp_table = pd.read_csv(f"tables/benign_long_{fp_dataset}.csv")
    fp_table = fp_table[(fp_table["defense"] == defense)]
    fp = fp_table.set_index("llm")["hard_FP"] / 100.0
    df = df.join(fp, on="llm")

    session_end_probs = df.apply(
        lambda row: session_end_probability(
            row["session_length"], row["hard_FP"], threshold
        ),
        axis=1,
    )
    return session_end_probs.mean()


def _preprocess_data():
    df = load_adaptive_defense_experiment()

    # add column that gives the prompt_id_in_session for the first success=True occurence
    # we have to assume that a session needs to be blocked by then, otherwise the attack was successful
    successes = df[df["success"] == True]
    first_success_idx = successes.groupby('session_id')['prompt_id_in_session'].first()
    first_success_idx.name = "first_success_in_session"
    df = df.join(first_success_idx, on="session_id")
    df["first_success_in_session"] = df["first_success_in_session"].fillna(-1).astype(int)

    # omit everything after the first success
    # this accounts for players who have potentially played multiple times
    df = df[df["prompt_id_in_session"] <= df["first_success_in_session"]]

    prompt_blocked = df.pivot_table(
        index=["setup", "llm", "prompt"],
        columns="defense",
        values="blocked",
        aggfunc="any",
    )
    prompt_blocked["D"] = prompt_blocked.any(axis=1)

    df.drop(columns=['defense', 'answer', 'raw_answer',
                     'blocked'], inplace=True)
    df.drop_duplicates(inplace=True)

    df = df.set_index(["setup", "llm", "prompt"])
    for defense in ["C1", "C2", "C3", "D"]:
        df[f"blocked_{defense}"] = prompt_blocked[defense]
        df[f"blocked_{defense}"] = df[f"blocked_{defense}"].astype(bool)
    df = df.reset_index()

    df_grouped = df.groupby(["setup", "llm", "session_id"])

    sess_df = pd.concat(
        [
            df_grouped["blocked_C1"].sum().rename("n_blocks_C1"),
            df_grouped["blocked_C2"].sum().rename("n_blocks_C2"),
            df_grouped["blocked_C3"].sum().rename("n_blocks_C3"),
            df_grouped["blocked_D"].sum().rename("n_blocks_D"),
            df_grouped["blocked_C1"].last().rename("last_blocked_C1"),
            df_grouped["blocked_C2"].last().rename("last_blocked_C2"),
            df_grouped["blocked_C3"].last().rename("last_blocked_C3"),
            df_grouped["blocked_D"].last().rename("last_blocked_D"),
            df_grouped["success"].any().rename("success"),
            df_grouped["prompt"].count().rename("session_length"),
        ],
        axis=1,
    )

    sess_df = sess_df.reset_index()

    # Only keep SUCCESSFUL session where last wasn't blocked by D
    sess_df = sess_df[(sess_df["success"]) & ~(sess_df["last_blocked_D"])]

    return sess_df


def block_threshold_defense_in_depth(
        fp_dataset: Literal["BasicUser", "BorderlineUser"] = "BasicUser",
        max_rounds: int = 20,
        llm_subplots: bool = False,
):
    sess_df = _preprocess_data()

    results = []
    for llm in sess_df.llm.unique():
        for defense in ["C1", "C2", "C3", "D"]:
            subset = sess_df[(sess_df["llm"] == llm)]
            for thres in list(range(1, max_rounds + 1, 1)):
                # for test_defense in [defense, "D"]:
                tpr = 1.0 - get_ssr(subset, defense, thres)
                fpr = get_fpr(fp_dataset, subset, defense, thres)
                tnr = 1.0 - fpr
                for kind, val in [("TPR", tpr), ("TNR", tnr)]:
                    results.append(
                        {
                            "llm": llm,
                            "defense": defense,
                            "threshold": thres,
                            "value": val,
                            "kind": kind,
                        }
                    )
    res = pd.DataFrame(results)

    if llm_subplots:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for i, llm in enumerate(res.llm.unique()):
            ax = axs[i]
            sns.lineplot(
                data=res[res["llm"] == llm],
                x="threshold",
                y="value",
                hue="defense",
                style="kind",
                ax=ax,
                errorbar=None,
            )
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_title(LLM_DISPLAY_NAMES[llm])
            ax.set_xlim(left=1)

            # only show legend in last panel
            if i != 2:
                ax.get_legend().remove()

    else:
        f = 0.9
        fig, ax = plt.subplots(figsize=(f * 6, f * 3.5), constrained_layout=True)
        ax = sns.lineplot(
            data=res,
            x="threshold",
            y="value",
            hue="defense",
            errorbar=None,
            style="kind",
        )
        ax.xaxis.set_major_locator(MultipleLocator(1))
        plt.xlim(left=1)

    label_dict = {
        "C1": "Substrings (C1)",
        "C2": "LLM check (C2)",
        "C3": "Strong prompt (C3)",
        "D": "Combined (D)",
        "defense": "Defense",
        "kind": "Metric",
        "TPR": "Security (AFR)",
        "TNR": "User Utility (SCR)",
    }

    handles, labels = plt.gca().get_legend_handles_labels()
    labels = [label_dict[l] if l in label_dict else l for l in labels]

    # remove where string is empty
    handles = [h for h, l in zip(handles, labels) if l != None]
    labels = [l for l in labels if l != None]

    plt.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.6), fontsize=8)

    plt.ylabel("AFR, SCR")
    plt.xlabel("Block Threshold")

    plt.ylim(bottom=0.0, top=1.0)

    plt.savefig(
        f"plots/session_level_defense_{fp_dataset}{'_sub' if llm_subplots else ''}.pdf"
    )
    plt.savefig(
        f"plots/session_level_defense_{fp_dataset}{'_sub' if llm_subplots else ''}.png"
    )
    plt.show()
    plt.close()


def block_threshold_developer_utility(
        fp_dataset: Literal["BasicUser", "BorderlineUser"] = "BasicUser",
        max_rounds: int = 20,
        llm_subplots: bool = False,
):
    sess_df = _preprocess_data()

    results = []
    defense = "D"
    for llm in tqdm(sess_df.llm.unique()):
        for defense in ["C1", "C2", "C3", "D"]:
            for lambda_ in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                subset = sess_df[(sess_df["llm"] == llm)]
                for thres in list(range(1, max_rounds + 1, 1)):
                    # for test_defense in [defense, "D"]:
                    tpr = 1.0 - get_ssr(subset, defense, thres)
                    fpr = get_fpr(fp_dataset, subset, defense, thres)
                    tnr = 1.0 - fpr

                    utility = (1 - lambda_) * tpr + lambda_ * tnr
                    results.append(
                        {
                            "llm": llm,
                            "defense": defense,
                            "threshold": thres,
                            "utility": utility,
                            "lambda": lambda_,
                        }
                    )
    res = pd.DataFrame(results)

    if llm_subplots:
        raise NotImplementedError()
    else:
        f = 0.9
        fig, ax = plt.subplots(figsize=(f * 6, f * 3.5), constrained_layout=True)
        sns.lineplot(
            data=res,
            x="threshold",
            y="utility",
            ax=ax,
            # hue="defense", style="lambda",
            hue="lambda",
            errorbar=None,
        )
        ax.xaxis.set_major_locator(MultipleLocator(1))
        plt.xlim(left=1)
        # plt.grid(axis="y")

    ax.legend(title="$\\lambda$", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.ylabel("Developer Utility")
    plt.xlabel("Block Threshold")

    plt.ylim(bottom=0.0, top=1.0)

    plt.title(fp_dataset)

    plt.savefig(f"plots/session_level_defense_devutil_{fp_dataset}.pdf")
    plt.show()
    plt.close()


def main():
    block_threshold_defense_in_depth(
        fp_dataset="BasicUser", max_rounds=20, llm_subplots=False
    )
    block_threshold_defense_in_depth(fp_dataset="BorderlineUser", max_rounds=20)

    block_threshold_developer_utility(fp_dataset="BasicUser", max_rounds=20)
    block_threshold_developer_utility(fp_dataset="BorderlineUser", max_rounds=20)


if __name__ == "__main__":
    main()
