from typing import Literal

import pandas as pd

from data import DEFENSE_DESCRIPTIONS, LLM_DISPLAY_NAMES, LLM_NAMES, load_benign_data
from analysis.utils import get_binomial_confidence_interval


def calculate_confidence_intervals(successes, total, confidence=0.95):
    lower, upper = get_binomial_confidence_interval(successes, total, confidence)
    return lower, upper


def get_pivot(result: pd.DataFrame, which: str):
    df = result[which].reset_index()
    df = df.pivot(columns="defense", index="llm", values=which)
    df.rename(columns=DEFENSE_DESCRIPTIONS, index=LLM_DISPLAY_NAMES, inplace=True)
    df = df[sorted(df.columns)]  # sort cols
    df = df.reindex(LLM_NAMES)  # sort rows
    df.index.name = "LLM"
    df.columns.name = None  # remove index name for prettier latex print
    return df.reset_index()


def accidental_reveals_table(result: pd.DataFrame, raw=False):
    col1 = "password_in_output_exact"
    col2 = "password_in_output"
    if raw:
        col1 = col1 + "_raw"
        col2 = col2 + "_raw"
    df = result[[col1, col2]].reset_index()

    df["accidental_reveal"] = (
            "\\makecell{"
            + df[col1].astype(str)
            + "\\\\{\\color{gray}("
            + df[col2].astype(str)
            + ")}}"
    )
    df = df.pivot(columns="defense", index="llm", values="accidental_reveal")
    df.rename(
        # columns=DEFENSE_DESCRIPTIONS,
        index=LLM_DISPLAY_NAMES,
        inplace=True,
    )
    df = df[sorted(df.columns)]  # sort cols
    df = df.reindex(LLM_NAMES)  # sort rows
    df.index.name = "LLM"
    df.columns.name = None  # remove index name for prettier latex print
    return df.reset_index()


def false_positives_tables(
        result: pd.DataFrame,
        conf_intervals: pd.DataFrame,
        which: Literal["soft", "hard"] = "hard",
):
    fp_rate = result[f"{which}_FP"]
    ci = conf_intervals[f"{which}_FP"]

    str_pattern = (
        lambda x, y: f"\\makecell{{{x}\\\\{{\\color{{gray}} "
                     f"\\tiny [{y[0] * 100:.2f}\%, {y[1] * 100:.2f}\%]}}}}"
    )

    df = fp_rate.combine(ci, str_pattern)
    df.name = "fp"
    df = df.reset_index()

    df = df.pivot(columns="defense", index="llm", values="fp")
    df.rename(
        # columns=DEFENSE_DESCRIPTIONS,
        index=LLM_DISPLAY_NAMES,
        inplace=True,
    )
    df = df[sorted(df.columns)]  # sort cols
    df = df.reindex(LLM_NAMES)  # sort rows
    df.index.name = "LLM"
    df.columns.name = None  # remove index name for prettier latex print

    df.drop(columns="A", inplace=True)

    return df.reset_index()


def generate_tables(dataset: Literal["BasicUser", "BorderlineUser"]):
    df = load_benign_data(dataset)

    # exclude data that was blocked in level A
    df = df[~df["is_level_a_block"]]

    # exclude Positives (reveals in any level)
    df = df[~df["is_reveal_exact"]]

    df["hard_FP"] = df["blocked"] & ~df["is_reveal"]
    df["soft_FP"] = df["blocked"] & ~df["is_reveal_exact"]

    grouped_df = df.groupby(["llm", "setup", "defense"])[
        [
            "blocked",
            "password_in_output",
            "password_in_output_exact",
            "password_in_output_subtle",
            "password_in_output_raw",
            "password_in_output_exact_raw",
            "password_in_output_subtle_raw",
            "hard_FP",
            "soft_FP",
        ]
    ]

    observations = grouped_df.count()
    positives = grouped_df.sum()

    # convert to percentage strings
    fractions = ((positives / observations) * 100)
    fractions.to_csv(f"tables/benign_long_{dataset}.csv")
    fractions = fractions.map(lambda x: f"{x:.1f}\%")

    ci = positives.copy()
    for col in positives.columns:
        ci[col] = positives[col].combine(
            observations[col],
            lambda successes, total: calculate_confidence_intervals(successes, total),
        )

    # save tables to latex
    accidental_reveals_table(fractions, raw=False).to_latex(
        f"tables/accidental_reveals_{dataset}.tex", index=False, column_format="c" * 8
    )

    false_positives_tables(fractions, ci).to_latex(
        f"tables/false_positives_{dataset}.tex", index=False, column_format="c" * 8
    )


def main():
    generate_tables("BasicUser")
    generate_tables("BorderlineUser")


if __name__ == "__main__":
    main()
