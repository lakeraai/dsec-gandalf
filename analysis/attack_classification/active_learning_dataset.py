from pathlib import Path
from typing import Literal

import pandas as pd
from ruamel.yaml import YAML

from data import load_rct_subsampled

from ..utils import compute_text_hash

LABELS_DIR = Path(__file__).parent / "labels"


def ellipsis(text: str, max_len: int = 100, unicode: bool = True) -> str:
    if len(text) > max_len:
        # JSON escapes unicode, so in JSON contexts it's useful to use "..." instead
        # of unicode "…".
        return text[:max_len] + ("…" if unicode else "...")
    return text


Split = Literal["train", "val", "test"]


def hash_to_split(hash: int) -> Split:
    fraction = (hash % 1000) / 1000  # good enough
    if fraction < 0.8:
        return "train"
    elif fraction < 0.9:
        return "val"
    else:
        return "test"


class ActiveLearningDataset:
    def __init__(self, category: str):
        self.category = category
        self.df = self.load_labels()

    def load_labels(self) -> pd.DataFrame:
        df = load_rct_subsampled()
        # deduplicate texts
        df = df.drop_duplicates(subset="prompt")

        df["label"] = None
        df["hash"] = df["prompt"].apply(compute_text_hash)
        df["split"] = df["hash"].apply(hash_to_split)
        df = df.set_index("hash")
        df = df.sort_index()

        if self.path_to_labels().exists():
            with open(self.path_to_labels(), "r") as f:
                d = YAML().load(f)

            for hash, sample_dict in d.items():
                # check that the hash is in the dataset
                if hash not in df.index:
                    raise ValueError(f"Hash {hash} not found in the dataset.")

                df.loc[hash, "label"] = sample_dict["label"]

        return df

    def save(self):
        d = {}
        for hash, row in self.df.iterrows():
            if row["label"] is not None:
                d[hash] = {
                    "label": row["label"],
                    "text_preview": ellipsis(row["prompt"]),
                }

        with open(self.path_to_labels(), "w") as f:
            YAML().dump(d, f)

    def path_to_labels(self) -> Path:
        return LABELS_DIR / f"{self.category}.yaml"

    def set_label(self, hash: int, label: bool):
        self.df.loc[hash, "label"] = label

    def split_by_labeled(self):
        mask = self.df["label"].notnull()
        labeled = self.df[mask]
        unlabeled = self.df[~mask]
        return labeled, unlabeled

    def get_random_seed(self) -> int:
        # Using a fixed random state for every category would also work but it makes
        # labeling a bit more interesting if we don't have to always label the same
        # samples at the beginning.
        return compute_text_hash(self.category) % 2**32
