from data import load_trial_data, LLM_DISPLAY_NAMES


def print_basic_statistics():
    df = load_trial_data()

    n_total_prompts = len(df[df["kind"] == "prompt"])
    n_sessions = len(df[["level", "llm", "user"]].drop_duplicates())
    n_users = len(df["user"].unique())

    print("N total prompts:", n_total_prompts)
    print("N sessions:", n_sessions)
    print("N users:", n_users)


def percentage_players_per_level(aggregate_c_levels: bool = True,
                                 use_solved: bool = True):
    """
    for each setup: compute the percentage of players that played a certain level
    """
    df = load_trial_data()

    if aggregate_c_levels:
        df["defense"] = df["defense"].map(lambda x: "C" if x.startswith("C") else x)

    players_total = df.groupby(["setup", "defense"])["user"].nunique()
    players_total = players_total.reset_index()

    if use_solved:
        level_solved = df.groupby(["setup", "llm", "user", "defense"])["success"].any()
        level_solved = level_solved.reset_index().groupby(["setup", "defense"])["success"].sum()
        players_total.set_index(["setup", "defense"], inplace=True)
        players_total["solved"] = level_solved
        players_total.reset_index(inplace=True)

    # relative fraction to defense A
    max_players = players_total.groupby(["setup"])["user"].max()  # level A
    max_players.name = "max_players"
    players_total = players_total.set_index("setup").join(max_players)

    if use_solved:
        players_total["fraction"] = players_total["solved"] / players_total["max_players"]
    else:
        players_total["fraction"] = players_total["user"] / players_total["max_players"]

    res = players_total.reset_index().pivot(index="setup", columns="defense", values="fraction")
    res.columns.name = None
    res.index.name = None
    res = res.applymap(lambda x: f"{x * 100.:.1f}\%")
    res.to_latex(f"tables/perc_players_per_level_{'solved' if use_solved else 'played'}.tex",
                 column_format="ccccc")

    print(res)


def avg_levels_solved():
    """
    for a fixed setup: create a table containing for each LLM the average of the last level each session managed to get to
    (i.e., if a session managed A and failed at B its a 1, if a session managed A and B but failed at all Cs its 2 and so on)
    This gives us a single number for how the LLMs affect the overall game difficulty.
    """
    df = load_trial_data()
    level_solved = df.groupby(["setup", "llm", "user", "defense"])["success"].any()
    level_solved = level_solved.reset_index()
    n_levels_solved = level_solved.groupby(["setup", "llm", "user"])["success"].sum()
    n_levels_solved = n_levels_solved.reset_index()
    avg_levels_solved = n_levels_solved.groupby(["setup", "llm"])["success"].mean()
    res = avg_levels_solved.reset_index().pivot(index="setup", columns="llm", values="success")
    res = res.rename(columns=LLM_DISPLAY_NAMES)
    res = res[["GPT-3.5-turbo", "GPT-4o-mini", "GPT-4"]]
    res.columns.name = None
    res.index.name = None
    res.to_latex(f"tables/avg_levels_solved.tex", float_format="%.2f", column_format="cccc")
    print(res)


def main():
    percentage_players_per_level(use_solved=True)
    avg_levels_solved()


if __name__ == "__main__":
    main()
