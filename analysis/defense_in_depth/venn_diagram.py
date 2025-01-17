from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib_venn import venn3
from matplotlib_set_diagrams import EulerDiagram, VennDiagram

from data import load_defense_in_depth_experiment, LLM_DISPLAY_NAMES

COMPUTER_MODERN_FONT = True

if COMPUTER_MODERN_FONT:
    rcParams['font.family'] = 'cmr10'
    rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titleweight'] = 'bold'


def prepare_dataset(save_file: bool = False):
    df = load_defense_in_depth_experiment()

    # exclude summarization for now as "blocked" estimation is unreliable
    df = df[df["setup"] != "summarization"]

    # only keep first trial_id, we do not analyze robustness yet
    df = df[df["trial_id"] == 0]

    got_blocked = df.groupby(["setup", "llm", "defense", "prompt"]).blocked.all()
    got_blocked = got_blocked.reset_index()

    df = got_blocked.pivot(
        columns="defense", index=["setup", "llm", "prompt"], values="blocked"
    )

    # exclude prompts that got blocked by level 2
    df = df[~df["B"]]

    df.drop(columns=["B"], inplace=True)
    df.columns.name = None
    df = df.reset_index()
    df.index.name = "prompt_id"

    if save_file:
        df.drop(columns="prompt").to_csv("analysis/defense_in_depth/data.csv")

    return df


def print_stats(df):
    print("N total: ", len(df))
    print("Blocks single levels:")
    print(df.sum(axis=0))
    print("Blocks combined: ", df.max(axis=1).sum(axis=0))


def venn_diag_1(df, ax=None, add_percentage: bool = True):
    # Calculate overlaps
    only_c1 = df["C1"] & ~df["C2"] & ~df["C3"]
    only_c2 = ~df["C1"] & df["C2"] & ~df["C3"]
    only_c3 = ~df["C1"] & ~df["C2"] & df["C3"]
    c1_and_c2 = df["C1"] & df["C2"] & ~df["C3"]
    c1_and_c3 = df["C1"] & ~df["C2"] & df["C3"]
    c2_and_c3 = ~df["C1"] & df["C2"] & df["C3"]
    all_three = df["C1"] & df["C2"] & df["C3"]

    # Get counts
    subsets = (
        only_c1.sum(),
        only_c2.sum(),
        c1_and_c2.sum(),
        only_c3.sum(),
        c1_and_c3.sum(),
        c2_and_c3.sum(),
        all_three.sum(),
    )

    # Create Venn diagram
    venn = venn3(subsets=subsets, set_labels=("C1", "C2", "C3"), ax=ax,
                 set_colors=("g", "b", "r"),
                 )

    if COMPUTER_MODERN_FONT:
        for label in venn.set_labels:
            if label:
                label.set_fontweight('bold')
                if ax is None:
                    label.set_fontsize(20)

        for subset_label in venn.subset_labels:
            if subset_label:
                subset_label.set_fontweight('bold')
                if ax is None:
                    subset_label.set_fontsize(20)

    if add_percentage:
        for label in venn.subset_labels:
            if label is not None:
                value = int(label.get_text())
                percentage = value / len(df) * 100
                label.set_text(
                    f"{percentage:.1f}%\n({value})")

    all_false = (~df["C1"] & ~df["C2"] & ~df["C3"]).sum()
    if add_percentage:
        percentage = all_false / len(df) * 100
        all_false_text = f"{percentage:.1f}%\n({all_false})"
    else:
        all_false_text = f"{all_false}"
    if ax is None:
        all_false_text += "\n(undetected)"

    (ax or plt).text(
        -0.6,
        -0.6,
        all_false_text,
        fontsize=10 if ax is not None else 20,
        color="black",
        horizontalalignment="center",
    )

    if ax is None:
        plt.title(f"n = {len(df)}", y=0.925, fontsize=20 if ax is None else None)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("plots/did_venn_1.pdf", bbox_inches="tight")
        plt.show()


def venn_subplots(df):
    setups = ["general",
              # "summarization",
              "topic"
              ]
    llms = ['openai_gpt-3.5-turbo-0125',
            'openai_gpt-4o-mini-2024-07-18',
            'openai_gpt-4-0125-preview',
            ]

    fig, axs = plt.subplots(len(setups), len(llms), figsize=(8, 6), constrained_layout=True)
    for i, setup in enumerate(setups):
        for j, llm in enumerate(llms):
            subset = df[(df.llm == llm) & (df.setup == setup)]

            venn_diag_1(subset, ax=axs[i][j])

            axs[i][j].axis('off')
            axs[i][j].set_title(f"{setup}, {LLM_DISPLAY_NAMES[llm]}, "
                                f"n={len(subset)}", fontsize=10, loc='center')

    plt.axis("off")
    plt.savefig("plots/did_venn_detailed_appendix.pdf")
    plt.close()


def main():
    df = prepare_dataset()

    venn_diag_1(df)
    venn_subplots(df)


if __name__ == '__main__':
    main()
