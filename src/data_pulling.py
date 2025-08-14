import nfl_data_py as nfl
import json
import pandas as pd


NGS_AGGREGATION_STRATEGY = {
    "sum": [],
    "average": ["avg_cushion", "avg_separation", "avg_intended_air_yards", "catch_percentage", 
                "avg_yac", "avg_expected_yac", "avg_yac_above_expectation", "avg_time_to_los", 
                "avg_rush_yards", "efficiency", "avg_time_to_throw", "avg_completed_air_yards"],
    "keep": ["season", "player_display_name", "player_position", "player_gsis_id"]
}


# weekly data (old)
# for season in seasons:
#     print(f"Downloading season {season}...")
#     season_data = nfl.import_weekly_data([season])

#     for position in positions:
#         pos_season_data = season_data[season_data["position"] == position].copy()

#         num_players = pos_season_data["player_display_name"].nunique()
#         print(f"{num_players} unique {position}s in {season}")

#         filename = f"data/raw/{position}_{season}_weekly.csv"
#         pos_season_data.to_csv(filename, index=False)
#         print(f"Saved to {filename}")
# print("Weekly data saved")


def pull_data(seasons=list(range(2000, 2025)), positions=["WR", "RB", "TE", "QB"]):
    print("SAVING DATA...")

    # all players at each relevant position
    roster_data = nfl.import_seasonal_rosters(seasons)
    players_dict = {
        "WR": set(),
        "RB": set(),
        "TE": set(),
        "QB": set()
    }
    for position in positions:
        pos_roster_data = roster_data[roster_data["position"] == position].copy()
        players_dict[position] = set(pos_roster_data["player_id"])
        print(f"Total {position}s: {len(players_dict[position])}")

    # seasonal data
    seasonal_data = nfl.import_seasonal_data(seasons)
    for position in positions:
        cleaned_seasonal_data = seasonal_data[seasonal_data["player_id"].isin(players_dict[position])].copy()
        filename = f"data/quarter_decade/{position}_seasonal_data.csv"
        cleaned_seasonal_data.to_csv(filename, index=False)
        print(f"{position} seasonal data saved")


    # player data
    for position in positions:
        cleaned_roster_data = roster_data[roster_data["player_id"].isin(players_dict[position])].copy()
        filename = f"data/quarter_decade/{position}_roster_data.csv"
        cleaned_roster_data.to_csv(filename, index=False)
        print(f"{position} roster data saved")


    # next-gen stats data
    ngs_data_receiving = nfl.import_ngs_data("receiving", seasons)
    ngs_data_receiving = aggregate_ngs_by_season(ngs_data_receiving)

    ngs_data_rushing = nfl.import_ngs_data("rushing", seasons)
    ngs_data_rushing = aggregate_ngs_by_season(ngs_data_rushing)

    ngs_data_passing = nfl.import_ngs_data("passing", seasons)
    ngs_data_passing = aggregate_ngs_by_season(ngs_data_passing)

    for position in positions:
        relevant_data = None

        # for QBs, consider both passing and rushing
        if position == "QB":
            relevant_data_1 = ngs_data_passing
            cleaned_relevant_data_1 = relevant_data_1[relevant_data_1["player_gsis_id"].isin(players_dict[position])].copy()
            filename1 = f"data/quarter_decade/{position}_ngs_data_passing.csv"
            cleaned_relevant_data_1.to_csv(filename1, index=False)

            relevant_data_2 = ngs_data_passing
            cleaned_relevant_data_2 = relevant_data_2[relevant_data_2["player_gsis_id"].isin(players_dict[position])].copy()
            filename2 = f"data/quarter_decade/{position}_ngs_data_rushing.csv"
            cleaned_relevant_data_2.to_csv(filename2, index=False)

        elif position == "WR" or position == "TE":
            relevant_data = ngs_data_receiving
        elif position == "RB":
            relevant_data = ngs_data_rushing

        if relevant_data is not None:  # skips over QBs
            cleaned_relevant_data = relevant_data[relevant_data["player_gsis_id"].isin(players_dict[position])].copy()
            filename = f"data/quarter_decade/{position}_ngs_data.csv"
            cleaned_relevant_data.to_csv(filename, index=False)

        print(f"{position} next-gen stats data saved")


def aggregate_ngs_by_season(ngs_dataframe, strategy_dict=NGS_AGGREGATION_STRATEGY):
    grouped = ngs_dataframe.groupby(['player_gsis_id', 'season'])

    aggregated_rows = []
    for (player_id, season), group_data in grouped:
        new_row = {}

        # "keep" columns
        for col in strategy_dict["keep"]:
            if col in group_data.columns:
                new_row[col] = group_data[col].iloc[0]
        
        # "sum" columns
        for col in strategy_dict["sum"]:
            if col in group_data.columns:
                new_row[col] = group_data[col].sum()

        # "average" columns
        for col in strategy_dict["average"]:
            if col in group_data.columns:
                new_row[col] = group_data[col].mean()

        new_row["total_weeks"] = len(group_data)

        aggregated_rows.append(new_row)

    return pd.DataFrame(aggregated_rows)


if __name__ == "__main__":
    pull_data()
