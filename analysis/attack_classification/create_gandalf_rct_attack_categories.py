import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

from analysis.attack_classification.predictions import get_df_with_categories
from analysis.embedding_utils import add_embeddings_column_to_df


def _get_last_attempt_for_group(df: pd.DataFrame) -> pd.Series:
    # Actually "success" is only set for guesses, but let's be explicit
    guesses = df.query("(kind == 'guess') & success")
    if guesses.empty:
        return df.iloc[-1]  # No guesses, take the last prompt
    else:
        return guesses.iloc[-1]  # Take the last guess


def get_last_attempt_for_df(df: pd.DataFrame) -> pd.DataFrame:
    tqdm.pandas()  # To be able to use the progress_apply method

    last_attempt = (
        df.groupby(["level", "user"])
        .progress_apply(_get_last_attempt_for_group, include_groups=False)  # takes ~30s
        .reset_index()
    )
    # Now we have a mix of "guess" and "prompt" rows. Normally "success" is not defined
    # for "prompts" but here the "prompt" rows correspond to (user, level) pairs without
    # a successful guess, so fill the "success" column.
    last_attempt["success"] = last_attempt["success"].fillna(False)
    return last_attempt


def main():
    """Get the last attempt for each (level, user), using the successful one
    if possible.

    Also adds embeddings to the dataset.
    """
    all_data = load_dataset("Lakera/gandalf-rct")["trial"].to_pandas()

    print("Extracting last attempt for each (level, user), this takes ~30s")
    df = get_last_attempt_for_df(all_data)

    df = add_embeddings_column_to_df(df, "text-embedding-3-small")
    df = get_df_with_categories(df)
    ds = Dataset.from_pandas(df)

    print(ds[0])
    print("^ First sample.")
    dataset_name = "Lakera/gandalf-rct-attack-categories"

    if input(f"Push dataset to {dataset_name}? (y/n): ").lower() != "y":
        print("Aborted.")
        exit(1)

    ds.push_to_hub(dataset_name, private=True)
    print(f"Pushed dataset to {dataset_name}")


if __name__ == "__main__":
    main()
