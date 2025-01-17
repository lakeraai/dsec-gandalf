import functools

import numpy as np
import openai
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from ..utils import compute_text_hash
from .active_learning_dataset import ActiveLearningDataset


@functools.cache
def get_embedding_with_cache(text: str) -> np.ndarray:
    client = openai.OpenAI()

    response = client.embeddings.create(model="text-embedding-3-small", input=[text])

    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def randomly(ds: ActiveLearningDataset) -> int:
    _, unlabeled = ds.split_by_labeled()
    unlabeled = unlabeled.sample(n=1000, random_state=ds.get_random_seed())
    return unlabeled.index[0]


def by_distance_from_labeled(ds: ActiveLearningDataset) -> int:
    labeled, unlabeled = ds.split_by_labeled()
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(labeled["embedding"].tolist())

    distances, _indices = nn.kneighbors(unlabeled["embedding"].tolist())
    return unlabeled.iloc[np.argmax(distances)].name


def by_similarity_to_example(ds: ActiveLearningDataset, positive_example: str) -> int:
    embedding = get_embedding_with_cache(positive_example)
    _, unlabeled = ds.split_by_labeled()
    unlabeled = unlabeled.copy()

    unlabeled["similarity"] = unlabeled["embedding"].apply(
        lambda e: cosine_similarity(embedding, e)
    )
    unlabeled = unlabeled.sort_values(by="similarity", ascending=False)

    return unlabeled.index[0]


def using_model(ds: ActiveLearningDataset, model: LogisticRegression) -> int:
    _, unlabeled = ds.split_by_labeled()
    random_state = compute_text_hash(ds.category) % 2**32
    unlabeled = unlabeled.sample(n=1000, random_state=random_state)

    prob_positive = model.predict_proba(unlabeled["embedding"].tolist())[:, 1]
    uncertainty = 1 - np.abs(prob_positive - 0.5) * 2
    return unlabeled.iloc[np.argmax(uncertainty)].name
