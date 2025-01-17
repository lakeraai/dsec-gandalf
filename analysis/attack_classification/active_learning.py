import argparse
import os
import platform

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import sample_selection
from .active_learning_dataset import ActiveLearningDataset


def clear_terminal():
    """Clear the terminal/console screen."""
    # Windows
    if platform.system().lower() == "windows":
        os.system("cls")
    # Unix/Linux/MacOS
    else:
        os.system("clear")


def get_key():
    try:
        # For Windows
        import msvcrt

        return msvcrt.getch().decode()
    except ImportError:
        # For Unix/Linux/MacOS
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def df_to_sklearn_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(df["embedding"].tolist())
    y = np.array(df["label"]).astype(int)
    return X, y


class NotEnoughData(Exception):
    """Not enough data to train a model."""


def train_model(
    ds: ActiveLearningDataset, respect_splits: bool = True
) -> LogisticRegression:
    """Train a logistic regression model on the labeled data.

    Setting `respect_splits` to False can be useful at the end once we're done with the
    labeling. We can always do cross-validation to get an estimate of the model's
    performance.
    """
    labeled, _ = ds.split_by_labeled()

    if respect_splits:
        # Only consider the train split here
        labeled = labeled.loc[labeled["split"] == "train"]

    n_positive = labeled["label"].sum()
    n_negative = len(labeled) - n_positive

    min_samples_per_label = 3
    if n_positive < min_samples_per_label or n_negative < min_samples_per_label:
        raise NotEnoughData(
            f"Need at least {min_samples_per_label} positives and negatives in the "
            "training set to train a model, "
            f"got {n_positive} positives and {n_negative} negatives."
        )

    model = LogisticRegression(C=10, class_weight="balanced")
    X, y = df_to_sklearn_dataset(labeled)
    model.fit(X, y)
    return model


def get_label_from_keyboard() -> bool | None:
    label = get_key().lower()
    if label == "q":
        raise KeyboardInterrupt

    yes_keys = {"y", "f"}
    no_keys = {"n", "d"}
    all_keys = yes_keys.union(no_keys)

    if label not in all_keys:
        clear_terminal()
        print(f"Invalid input. Please enter one of {all_keys.union('q')}.")
        return None

    is_yes = label in yes_keys
    return is_yes


def main(category: str, exploration_prob: float, positive_example: str | None = None):
    print("Loading...")
    ds = ActiveLearningDataset(category)
    while True:
        labeled, _ = ds.split_by_labeled()
        if positive_example:
            model = None
            hash = sample_selection.by_similarity_to_example(ds, positive_example)
        else:
            try:
                model = train_model(ds)
                if np.random.rand() < exploration_prob:
                    hash = sample_selection.randomly(ds)
                else:
                    hash = sample_selection.using_model(ds, model)
            except NotEnoughData:
                hash = sample_selection.randomly(ds)
                model = None

        print(f"Category: {category}")
        print(
            f"Labeled samples: {len(labeled)} "
            f"({labeled['label'].sum()} positive, "
            f"{len(labeled) - labeled['label'].sum()} negative)"
        )
        if model is not None:
            val_samples = labeled.loc[labeled["split"] == "val"]
            if len(val_samples) > 0:
                X, y = df_to_sklearn_dataset(val_samples)
                print(f"Val accuracy: {model.score(X, y):.2f} (n={len(X)})")

        print(f"Is this under category {category}? (n/d=no, y/f=yes)")
        print()
        print(ds.df.loc[hash, "prompt"])
        try:
            label = get_label_from_keyboard()
            if label is None:
                continue
        except KeyboardInterrupt:
            print("Quitting.")
            break

        ds.set_label(hash, label)
        ds.save()
        clear_terminal()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("category", help="The category to label.")
    parser.add_argument(
        "--exploration-prob",
        type=float,
        help="Even if we can train a model, use random selection for some of the "
        "prompts to do more exploration. Should be in [0, 1]",
    )
    parser.add_argument(
        "--positive-example",
        type=str,
        default=None,
        help="If provided, find prompts similar to this one.",
    )
    args = parser.parse_args()

    exploration_prob = args.exploration_prob

    if exploration_prob is not None and args.positive_example is not None:
        raise ValueError(
            "Can't provide both --exploration-prob and --positive-example."
        )

    if exploration_prob is None:
        exploration_prob = 0.2

    main(
        args.category,
        exploration_prob=exploration_prob,
        positive_example=args.positive_example,
    )
