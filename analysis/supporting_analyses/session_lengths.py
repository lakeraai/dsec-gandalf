"""
Script to create bar chart plots to analyze session lengths
and successes w.r.t. setup, defender, and LLM name.
"""

from typing import Literal

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

from data import (
    DEFENSE_DESCRIPTIONS,
    LLM_DISPLAY_NAMES,
    TRIAL_DEFENSES,
    load_trial_data,
)

COMPUTER_MODERN_FONT = True
if COMPUTER_MODERN_FONT:
    rcParams['font.family'] = 'cmr10'
    rcParams['axes.formatter.use_mathtext'] = True

LLM_NAMES = list(LLM_DISPLAY_NAMES.keys())

INDEX_COLS = ["setup", "llm", "defense"]

# Define custom color palette with blue as C0 and green as C1
custom_palette = ["#DC3220", "#1E88E5"]
sns.set_palette(custom_palette)


def prepare_data():
    df = load_trial_data()

    is_success = df.groupby(INDEX_COLS + ["user"])["success"].any()

    df_prompts_only = df[df["kind"] == "prompt"]
    n_trials = df_prompts_only.groupby(INDEX_COLS + ["user"])["success"].apply(len)

    df2 = pd.DataFrame()
    df2["success"] = is_success
    df2["n_trials"] = n_trials
    df2.reset_index(inplace=True)

    return df2


def plot_panel(
        df: pd.DataFrame,
        setup: Literal["general", "summarization", "topic", "all"],
        clip_value: int = 60,
        include_defenses=("A", "B", "C1", "C2", "C3", "D"),
        add_rectangle: bool = False,
):
    assert clip_value == 60, "Not implemented yet"

    if clip_value:
        df["n_trials"] = df["n_trials"].clip(upper=clip_value)

    w = 16 / 6 * len(include_defenses)
    h = 7 if setup == "all" else 8

    f = 0.8
    w *= f
    h *= f
    figsize = (w, h)

    fig, axs = plt.subplots(
        len(LLM_NAMES),
        len(include_defenses),
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        sharey=False,
    )

    for i, llm_name in enumerate(LLM_NAMES):
        for j, defense in enumerate(include_defenses):
            subset = df[(df["llm"] == llm_name) & (df["defense"] == defense)]

            if setup != "all":
                subset = subset[(subset["setup"] == setup)]

            subset_pivot = subset.pivot(
                columns="success", values="n_trials", index=["user", "setup"]
            )
            subset_pivot.rename(
                columns={False: "no success", True: "success"}, inplace=True
            )

            if len(subset_pivot) < 3:
                continue

            sns.histplot(
                data=subset_pivot,
                multiple="stack",
                binwidth=3,
                kde=False,
                line_kws={"linewidth": 2},
                alpha=1.0,
                common_norm=False,
                ax=axs[i, j],
                stat="count",
            )

            if i == 0:
                axs[i, j].set_title(DEFENSE_DESCRIPTIONS[defense])
            if i == len(LLM_NAMES) - 1:
                axs[i, j].set_xlabel("# transactions")

            if j == 0:
                axs[i, j].set_ylabel(f"{LLM_DISPLAY_NAMES[llm_name]}\nCount")
            else:
                axs[i, j].set_ylabel(None)

            if i == 0 and j == len(TRIAL_DEFENSES) - 1:
                axs[i, j].get_legend().set_title("")
            else:
                axs[i, j].get_legend().remove()

            axs[i, j].yaxis.set_major_locator(MaxNLocator(integer=True))

    # set greater equal sign for clipped values
    j = 0
    tick_positions = [0, 20, 40, 60]
    tick_labels = [0, 20, 40, "$≥$60"]
    axs[-1, j].set_xticks(tick_positions)
    axs[-1, j].set_xticklabels(tick_labels)

    if add_rectangle:
        x1, x2, color = (0.357, 0.843, "grey")

        rect = patches.Rectangle(
            (x1, 0.0),
            x2 - x1,
            0.970,
            transform=fig.transFigure,
            color=color,
            alpha=0.25,
            zorder=-1,
        )
        fig.patches.append(rect)

    if setup != "all":
        plt.suptitle(f"Setup: {setup}", fontweight="bold")
    plt.savefig(f"plots/session_len_{setup}.pdf")
    plt.show()
    plt.close()


def main():
    data = prepare_data()
    plot_panel(data, "general", add_rectangle=True)
    plot_panel(data, "summarization", add_rectangle=True)
    plot_panel(data, "topic", add_rectangle=True)


if __name__ == "__main__":
    main()
