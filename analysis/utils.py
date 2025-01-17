import hashlib

from scipy import stats
import numpy as np


def compute_text_hash(text: str) -> int:
    """Computes a hash for a given text and returns it as an integer.

    Args:
        text: The text to hash.

            Returns:
        The hash of the text as an integer.
    """
    hashed = hashlib.md5(text.encode("utf-16")).hexdigest()
    return int(hashed, 16)


def get_binomial_confidence_interval(successes, total, confidence=0.95):
    alpha = 1 - confidence
    half_alpha = alpha / 2

    lower = stats.beta.ppf(half_alpha, successes, total - successes + 1)
    upper = stats.beta.ppf(1 - half_alpha, successes + 1, total - successes)

    if np.isnan(lower):
        lower = 0

    if np.isnan(upper):
        upper = 1

    return lower, upper
