from pathlib import Path
from typing import Literal

import diskcache
import numpy as np
import openai
import pandas as pd
from datasets import Dataset

from analysis.utils import compute_text_hash

EmbeddingModel = Literal["text-embedding-3-small", "text-embedding-3-large"]


def get_embeddings(texts: list[str], model: EmbeddingModel) -> list[np.ndarray]:
    client = openai.OpenAI()

    # Not sure why, but sometimes this gets stuck for a really long time, like 10 min.
    response = client.embeddings.create(model=model, input=texts)

    embeddings = [np.array(d.embedding, dtype=np.float32) for d in response.data]

    return embeddings


def add_embeddings_column(ds: Dataset | pd.Series, model: EmbeddingModel) -> Dataset:
    if isinstance(ds, pd.Series):
        df = pd.DataFrame(ds)
        df.rename(columns={df.columns[0]: "prompt"}, inplace=True)
        ds = Dataset.from_pandas(df)

    def add_embeddings(samples: dict[str, list]):
        # Naive caching - caches on the level of the whole batch
        embeddings_cache = diskcache.Cache(Path(__file__).parent / "embeddings_cache")
        key = model + "/" + str(compute_text_hash(str(samples["prompt"])))

        if key in embeddings_cache:
            samples["embedding"] = embeddings_cache[key]
        else:
            samples["embedding"] = get_embeddings(samples["prompt"], model)
            embeddings_cache[key] = samples["embedding"]

        return samples

    with_embeddings = ds.map(
        add_embeddings, batched=True, batch_size=100, desc="Retrieving text embeddings"
    )
    return with_embeddings


def add_embeddings_column_to_df(
    df: pd.DataFrame, model: EmbeddingModel
) -> pd.DataFrame:
    ds = Dataset.from_pandas(df)
    ds = add_embeddings_column(ds, model)
    return ds.to_pandas()
