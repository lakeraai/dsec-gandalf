import numpy as np
import pandas as pd

from data import load_defense_in_depth_experiment, load_benign_data
from itertools import product


def prepare_dataset(benign_name):
    # Attacker data
    df = load_defense_in_depth_experiment()
    # exclude summarization for now as "blocked" estimation is unreliable
    df = df[df["setup"] != "summarization"]
    df = df[df["trial_id"] == 0]
    got_blocked = df.groupby(["setup", "llm", "defense", "prompt"]).blocked.all()
    got_blocked = got_blocked.reset_index()
    df = got_blocked.pivot(
        columns="defense", index=["setup", "llm", "prompt"], values="blocked"
    )
    # exclude prompts that got blocked by level 2
    df = df[~df["B"]]
    df.drop(columns=["B"], inplace=True)
    df.columns.name = None
    df = df.reset_index()
    df.index.name = "prompt_id"
    df["data_type"] = "attacker"
    # User data
    df2 = load_benign_data(benign_name)
    # exclude summarization for now as "blocked" estimation is unreliable
    df2 = df2[df2["setup"] == "vanilla"]
    got_blocked = df2.groupby(["setup", "llm", "defense", "prompt"]).blocked.all()
    got_blocked = got_blocked.reset_index()
    df2 = got_blocked.pivot(
        columns="defense", index=["setup", "llm", "prompt"], values="blocked"
    )
    # exclude prompts that got blocked by level 2
    df2 = df2[~df2["B"]]
    df2.drop(columns=["A", "B", "D"], inplace=True)
    df2.columns.name = None
    df2 = df2.reset_index()
    df2.index.name = "prompt_id"
    df2["data_type"] = "user"
    # Combine datasets
    df = pd.concat([df, df2], ignore_index=True)
    return df


def compute_block(df):
    combinations = list(product([True, False], repeat=3))
    block_rates = {}
    for data_type in ["attacker", "user"]:
        block_rates[data_type] = {}
        for comb in combinations:
            c1 = df[df["data_type"] == data_type]["C1"] == comb[0]
            c2 = df[df["data_type"] == data_type]["C2"] == comb[1]
            c3 = df[df["data_type"] == data_type]["C3"] == comb[2]
            block_rates[data_type][comb] = (c1 & c2 & c3).mean()
    return block_rates


def optimize_aggreation(block_rates, tradeoff_pars):
    combinations = list(product([True, False], repeat=3))
    aggregation_functions = list(product([0, 1], repeat=len(combinations)))
    utility = []
    security = []
    for aggregation_fun in aggregation_functions:
        security.append(
            np.sum(
                [
                    block_rates["attacker"][comb]
                    for ii, comb in enumerate(combinations)
                    if aggregation_fun[ii] == 1
                ]
            )
        )
        utility.append(
            np.sum(
                [
                    block_rates["user"][comb]
                    for ii, comb in enumerate(combinations)
                    if aggregation_fun[ii] == 0
                ]
            )
        )
    or_index = -2
    and_index = [
        ii
        for ii, agg_fun in enumerate(aggregation_functions)
        if agg_fun == (1, 0, 0, 0, 0, 0, 0, 0)
    ][0]
    result_matrix = np.zeros((len(tradeoff_pars), 3))
    opt_aggregations = []
    for ii, tradeoff_par in enumerate(tradeoff_pars):
        developer_utility = (1 - tradeoff_par) * np.array(
            security
        ) + tradeoff_par * np.array(utility)
        opt_index = np.argmax(developer_utility)
        result_matrix[ii, 0] = developer_utility[or_index]
        result_matrix[ii, 1] = developer_utility[and_index]
        result_matrix[ii, 2] = developer_utility[opt_index]
        opt_aggregations.append(aggregation_functions[opt_index])
    # find optimal aggregation
    return result_matrix, opt_aggregations


def main():
    df = prepare_dataset("BorderlineUser")
    block_rates = compute_block(df)
    result_matrix, opt_aggregations = optimize_aggreation(block_rates, [0, 1 / 4, 1 / 2, 3 / 4, 1])
    print(result_matrix)
    print(opt_aggregations)


if __name__ == '__main__':
    main()
