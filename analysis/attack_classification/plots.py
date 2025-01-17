from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from analysis.attack_classification.predictions import (
    CATEGORY_DISPLAY_NAMES,
    get_categories_from_df,
    get_df_with_categories,
)

from ..utils import get_binomial_confidence_interval

SETUPS = ["general", "summarization", "topic"]

PLOTS_DIR = Path(__file__).parents[2] / "plots"


def make_detailed_attack_categories_plot(success_rates: pd.DataFrame):
    success_rates["attack_category"] = success_rates["attack_category"].map(
        CATEGORY_DISPLAY_NAMES
    )
    fig = px.bar(
        success_rates,
        x="defense",
        y="success_rate",
        color="attack_category",
        facet_row="setup",
        width=1000,
        height=600,
        barmode="group",
        hover_data={
            "n_attacks": True,
            "fraction_of_level": True,
        },
        error_y="error_upper",
        error_y_minus="error_lower",
        labels={
            "defense": "Defense",
            "success_rate": "Success rate",
            "attack_category": "Attack category",
            "setup": "Setup",
        },
    )

    return fig


def analyze_attack_success_rates(df: pd.DataFrame) -> pd.DataFrame:
    # Get all binary prediction columns
    binary_cols = [col for col in df.columns if col.startswith("binary_prediction_")]

    # Group by level and calculate success rates for each attack category
    results = []

    for (level_name, setup, defense), level_group in df.groupby(
        ["level", "setup", "defense"]
    ):
        level_size = len(level_group)

        for col in binary_cols:
            attack_category = col[len("binary_prediction_") :]

            attack_attempts = level_group[level_group[col] == True]  # noqa: E712

            if attack_attempts.empty:
                continue

            n_attacks = len(attack_attempts)
            n_successful = len(attack_attempts[attack_attempts["success"] == True])  # noqa: E712

            ci_lower, ci_upper = get_binomial_confidence_interval(
                n_successful, n_attacks
            )
            success_rate = n_successful / n_attacks
            assert ci_lower <= success_rate <= ci_upper, (
                f"Success rate {success_rate} not in confidence interval "
                f"({ci_lower}, {ci_upper}) for level {level_name}"
            )

            results.append(
                {
                    "level": level_name,
                    "setup": setup,
                    "defense": defense,
                    "attack_category": attack_category,
                    "n_attacks": n_attacks,
                    "success_rate": success_rate,
                    "error_lower": success_rate - ci_lower,
                    "error_upper": ci_upper - success_rate,
                    "fraction_of_level": (n_attacks / level_size),
                }
            )

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["level", "success_rate"], ascending=[True, False]
    )

    # Round numeric columns for better readability
    results_df["success_rate"] = results_df["success_rate"].round(3)
    results_df["fraction_of_level"] = results_df["fraction_of_level"].round(3)

    return results_df


def get_category_success_rate(
    df: pd.DataFrame,
    category: str,
    split_on: list[str],
    verbose: bool = False,
) -> float:
    """Get the average success rate for people using a given attack category."""
    to_average = []

    for group_key, group_prompts in df.groupby(split_on):
        uses_attack = group_prompts[f"binary_prediction_{category}"]

        attack_success_rate = group_prompts.loc[uses_attack, "success"].mean()

        if verbose:
            print(f"{group_key:>20} {attack_success_rate:.2f}")

        # NaN happens if there are no attacks of this category
        # (or no attacks _without_ this category, but that is unlikely)
        if not np.isnan(attack_success_rate):
            to_average.append(
                {
                    "attack_success_rate": attack_success_rate,
                    "weight": len(group_prompts),
                }
            )
        else:
            print(f"No attacks of category {category} in {group_key}")

    to_average = pd.DataFrame(to_average)
    average = np.average(
        to_average["attack_success_rate"], weights=to_average["weight"]
    )
    return average


def get_category_success_rate_df(df: pd.DataFrame, split_on: list[str]):
    effects_df = []

    for category in get_categories_from_df(df):
        for setup in SETUPS:
            success_rate = get_category_success_rate(
                df.loc[df["setup"] == setup], category, split_on=split_on
            )
            effects_df.append(
                {
                    "attack_category": category,
                    "setup": setup,
                    "success_rate": success_rate,
                }
            )

    return pd.DataFrame(effects_df)


def get_category_effect_plot(df: pd.DataFrame):
    effects_df = get_category_success_rate_df(df, split_on=["level", "llm"])
    effects_df = effects_df.set_index(["attack_category", "setup"])

    # For input obfuscation, there is not enough data to adjust for the LLM as well.
    # We use a simpler approach in which the LLM is not considered. We observed that
    # this has little effect on the results in other attack categories.
    effects_df_no_llm = get_category_success_rate_df(df, split_on=["level"])
    effects_df_no_llm = effects_df_no_llm.set_index(["attack_category", "setup"])
    assert effects_df.index.equals(effects_df_no_llm.index)
    mask = [category == "input_obfuscation" for category, _ in effects_df.index]
    effects_df.loc[mask, "success_rate"] = effects_df_no_llm.loc[mask, "success_rate"]
    effects_df = effects_df.reset_index()

    # Mark it with an asterisk to indicate that the column is different
    effects_df["attack_category"] = effects_df["attack_category"].map(
        {**CATEGORY_DISPLAY_NAMES, "input_obfuscation": "Input obfuscation*"}
    )

    return px.bar(
        effects_df,
        x="attack_category",
        y="success_rate",
        color="setup",
        barmode="group",
        labels={
            "success_rate": "Interventional Attacker Success Rate",
            "setup": "Setup",
            "attack_category": "Attack category",
        },
        range_y=[0, 1],
    )


def save_attack_categories_plots():
    # Save a dummy image as a workaround for
    # https://github.com/plotly/plotly.py/issues/3469
    fig = px.scatter([{"x": 0, "y": 0}])
    fig.write_image("/tmp/dummy.pdf")

    df = get_df_with_categories()
    success_rates = analyze_attack_success_rates(df)

    def save(fig, name):
        fig.write_image(PLOTS_DIR / f"{name}.pdf")
        print(f"Saved plot to {PLOTS_DIR / name}.pdf")

    fig_detailed = make_detailed_attack_categories_plot(success_rates)
    save(fig_detailed, "attack_categories_detailed")

    fig_effect = get_category_effect_plot(df.query('defense != "D"'))
    save(fig_effect, "attack_categories_effect")

    return fig_detailed, fig_effect


if __name__ == "__main__":
    save_attack_categories_plots()
