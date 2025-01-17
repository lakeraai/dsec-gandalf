"""
Analyse output quality degradation for benign prompts besides model refusal.

Compare Levels A, B, and C3 for Ultrachat prompts.

Using the same LLM, how does the output change?

Two metrics:

- string lengths
- 1 / relative cosine similarity compared to level A
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from openai import OpenAI
from scipy.stats import wilcoxon
from tqdm import tqdm

from analysis.embedding_utils import EmbeddingModel, add_embeddings_column
from data import load_benign_data

COMPUTER_MODERN_FONT = True

if COMPUTER_MODERN_FONT:
    rcParams["font.family"] = "cmr10"
    rcParams["axes.formatter.use_mathtext"] = True

tqdm.pandas()

client = OpenAI()


def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)


def boxplots(openai_embeddings: EmbeddingModel = "text-embedding-3-large"):
    df = load_benign_data("BasicUser")

    df = df[df["llm"] == "openai_gpt-4-0125-preview"]

    # only keep system prompt-based defenses
    df = df[df["defense"].isin(["A", "B", "C3"])]

    # only compare prompts where no defense blocked
    blocked_by_one_defense = df[df["blocked"]][["llm", "prompt_id"]]
    blocked_by_one_defense_set = set(
        zip(blocked_by_one_defense["llm"], blocked_by_one_defense["prompt_id"])
    )
    df["blocked_by_one"] = df.apply(
        lambda row: (row["llm"], row["prompt_id"]) in blocked_by_one_defense_set, axis=1
    )
    df = df[~df["blocked_by_one"]]

    df["n_chars"] = df["answer"].map(len)

    ds_with_embeddings = add_embeddings_column(df["answer"], openai_embeddings)
    # Use .values because the index is different (but we know it matches)
    df["emb"] = ds_with_embeddings.to_pandas()["embedding"].values

    # pivot to defense-per-column format
    dfp_chars = df.pivot(
        index=["llm", "prompt_id"], columns="defense", values="n_chars"
    ).reset_index(drop=True)
    dfp_chars_rel = dfp_chars.divide(dfp_chars["A"], axis=0)

    dfp_embs = df.pivot(
        index=["llm", "prompt_id"], columns="defense", values="emb"
    ).reset_index(drop=True)
    dfp_cossims = dfp_embs.copy()
    for col in dfp_cossims.columns:
        dfp_cossims[col] = dfp_embs.apply(
            lambda row: cosine_similarity(row["A"], row[col]), axis=1
        )

    f = 0.7
    fig, ax = plt.subplots(1, 2, figsize=(f * 6, f * 3), constrained_layout=True)

    meanprops = {
        #    'marker': 'v',          # Triangle marker
        "markerfacecolor": "red",  # Triangle fill color
        "markeredgecolor": "red",  # Triangle border color
        #    'markersize': 10        # Size of the triangle
    }

    val_name = "Relative Length"
    ax[0].set_title("Relative prompt length (compared to A)", fontsize=9)
    plot_data = dfp_chars_rel[["B", "C3"]].melt(value_name=val_name)
    x = sns.boxplot(
        plot_data,
        x="defense",
        y=val_name,
        showfliers=False,
        showmeans=True,
        ax=ax[0],
        meanprops=meanprops,
        boxprops=dict(facecolor="white", edgecolor="black"),
    )
    ax[0].set_xlabel(None)

    # get p-values for left subplot (one-sided Wilcoxom Rank Sum)
    p_B = wilcoxon(
        dfp_chars["A"],
        dfp_chars["B"],
        alternative="greater",
        method="exact",
        zero_method="wilcox",
    )
    p_C3 = wilcoxon(
        dfp_chars["A"],
        dfp_chars["C3"],
        alternative="greater",
        method="exact",
        zero_method="wilcox",
    )
    print("P-values: ", p_B.pvalue, p_C3.pvalue)

    ax[1].set_title("Embedding Similarity with A", fontsize=9)
    val_name = "Cosine Similarity"
    sns.boxplot(
        dfp_cossims[["B", "C3"]].melt(value_name=val_name),
        x="defense",
        y=val_name,
        showfliers=False,
        showmeans=True,
        ax=ax[1],
        boxprops=dict(facecolor="white", edgecolor="black"),
        meanprops=meanprops,
    )
    ax[1].set_xlabel(None)

    plt.savefig("plots/benign_output_quality.pdf")
    plt.show()
    plt.close()


def main():
    boxplots(openai_embeddings="text-embedding-3-large")


if __name__ == "__main__":
    main()
