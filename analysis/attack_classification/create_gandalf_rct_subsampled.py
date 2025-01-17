"""Create/update Lakera/gandalf-rct-subsampled on Hugging Face Hub.

Uses Lakera/gandalf-trial-data as a source, it takes N_SAMPLES_PER_LEVEL samples per
level. Doesn't only look at successful guesses, any row with kind == "prompt" get be
sampled.

Limits the number of prompts per user to MAX_PROMPTS_PER_user. This is because a single
user often uses a lot of variants of a single prompt, so this limits the diversity of
the data.

Adds text embeddings to the samples (OpenAI's text-embedding-3-small). We need the
embeddings to classify the attack approaches.
"""

import functools
from collections import defaultdict

import tqdm.auto
from datasets import Dataset, Features, IterableDataset, load_dataset

from analysis.attack_classification.embedding_utils import add_embeddings_column

N_SAMPLES_PER_LEVEL = 1000
# The lower this limit is, the higher the diversity of the data should be
MAX_PROMPTS_PER_LEVEL_AND_USER = 2
# Levels with less data
MAX_PROMPTS_OVERRIDES = {
    "topic-D": 5,
    "general-D": 10,
    "summarization-D": N_SAMPLES_PER_LEVEL,  # No limit, very few samples
}


def get_level_names(ds: Dataset) -> set[str]:
    """A lazy way to get the level names from the dataset.

    Also, if we rename the levels we don't have to change this function because we don't
    rely on the specific names.
    """
    EXPECTED_N_LEVELS = 18
    levels = set()
    for sample in ds:
        levels.add(sample["level"])
        if len(levels) == EXPECTED_N_LEVELS:
            break

    return levels


def dataset_from_iterable_dataset(ds: IterableDataset, features: Features) -> Dataset:
    """Convert an IterableDataset to a Dataset.

    Weirdly, `datasets` doesn't have this built in.
    """

    def gen_from_iterable_dataset(iterable_ds: IterableDataset):
        yield from iterable_ds

    return Dataset.from_generator(
        functools.partial(gen_from_iterable_dataset, ds), features=features
    )


def sample_equally_from_levels(ds: Dataset, samples_per_level: int):
    """Sample equally from each level in the dataset."""
    level_names = get_level_names(ds)
    samples = []
    n_samples_collected = {level: 0 for level in level_names}

    ds = ds.shuffle(seed=37)  # Shuffle the dataset to get a random sample

    n_prompts_per_level_and_user = defaultdict(int)
    n_prompts_refused = 0

    for sample in (
        progress_bar := tqdm.auto.tqdm(
            ds, desc=f"Sampling {samples_per_level} prompts per level"
        )
    ):
        level = sample["level"]
        user = sample["user"]

        n_levels_done = sum(
            n_samples_collected[level] == samples_per_level for level in level_names
        )
        if n_prompts_per_level_and_user[(level, user)] >= MAX_PROMPTS_OVERRIDES.get(
            level, MAX_PROMPTS_PER_LEVEL_AND_USER
        ):
            n_prompts_refused += 1
            progress_bar.set_postfix(
                {"n_prompts_refused": n_prompts_refused, "n_levels_done": n_levels_done}
            )
            continue
        n_prompts_per_level_and_user[(level, user)] += 1

        if n_samples_collected[level] < samples_per_level:
            samples.append(sample)
            n_samples_collected[level] += 1

        if n_levels_done == len(level_names):
            break

    for level in level_names:
        if n_samples_collected[level] < samples_per_level:
            raise ValueError(
                f"Could not sample enough samples from level {level}. "
                f"Only got {n_samples_collected[level]} samples, "
                f"expected {samples_per_level}."
            )

    ds = Dataset.from_list(samples)
    ds = ds.sort(["level", "datetime"])
    return ds


def main():
    ds = load_dataset("Lakera/gandalf-trial-data")["trial"]
    # Can't get embeddings for empty prompts
    ds = ds.filter(lambda x: x["prompt"] != "" and x["kind"] == "prompt")

    ds = sample_equally_from_levels(ds, N_SAMPLES_PER_LEVEL)
    ds = add_embeddings_column(ds, "text-embedding-3-small")

    print(ds[0])
    print("^ First sample.")
    dataset_name = "Lakera/gandalf-rct-subsampled"

    if input(f"Push dataset to {dataset_name}? (y/n): ").lower() != "y":
        print("Aborted.")
        exit(1)

    ds.push_to_hub(dataset_name, private=True)
    print(f"Pushed dataset to {dataset_name}")


if __name__ == "__main__":
    main()
