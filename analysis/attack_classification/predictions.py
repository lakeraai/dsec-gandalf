import pandas as pd
import tqdm.auto
from datasets import load_dataset

from .active_learning import ActiveLearningDataset, train_model

CATEGORY_DISPLAY_NAMES = {
    "direct": "Direct",
    "partial": "Partial",
    "indirect": "Indirect",
    "input_obfuscation": "Input obfuscation",
    "output_obfuscation": "Output obfuscation",
    "output_fiction": "Output fiction",
    "input_non_english": "Non-English",
    # "output_non_english": "Output non-English", # turned out to be very rare
    "persuasion": "Persuasion",
    "context_override": "Context override",
}

CATEGORIES = list(CATEGORY_DISPLAY_NAMES.keys())


def add_predictions_for_category(df: pd.DataFrame, category: str):
    """Add a prediction_{category} column in-place to the dataframe."""
    ald = ActiveLearningDataset(category)
    model = train_model(ald, respect_splits=False)

    prob_positive = model.predict_proba(df["embedding"].tolist())[:, 1]
    df[f"prediction_{category}"] = prob_positive
    df[f"binary_prediction_{category}"] = prob_positive > 0.5


def get_categories_from_df(df: pd.DataFrame):
    """Get the categories from a DataFrame that has prediction columns.

    It's a bit more robust to compute categories dynamically here rather than using
    CATEGORIES in case we change `df` somehow. This only returns actually present
    categories.
    """
    return [c[len("prediction_") :] for c in df.columns if c.startswith("prediction_")]


def add_salient_category_column(df: pd.DataFrame):
    categories = get_categories_from_df(df)
    salient_category = df[[f"prediction_{c}" for c in categories]].idxmax(axis=1)
    max_prediction = df[[f"prediction_{c}" for c in categories]].max(axis=1)

    # strip the prediction_ prefix
    salient_category = salient_category.str[len("prediction_") :]

    # if the max is too low, set to None
    salient_category[max_prediction < 0.6] = None

    df["salient_category"] = salient_category

    return salient_category


def get_df_with_categories(base_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if base_df is None:
        # A bit redundant because this dataset already has attack categories,
        # but can be useful if we change the classifiers.
        ds = load_dataset("Lakera/gandalf-rct-attack-categories")["train"]
        df = ds.to_pandas()
    else:
        df = base_df.copy()

    for category in tqdm.auto.tqdm(
        CATEGORIES, desc="Adding attack categories predictions"
    ):
        add_predictions_for_category(df, category)

    add_salient_category_column(df)

    return df
