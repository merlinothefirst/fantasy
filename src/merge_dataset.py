import pandas as pd
import json


TEAM_NAME_TO_ABBREV = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",       # formerly OAK
    "Oakland Raiders": "OAK",        # pre-2020
    "Los Angeles Chargers": "LAC",   # formerly SD
    "San Diego Chargers": "SD",      # pre-2017
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",  # formerly Football Team / Redskins
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS"
}



def merge_datasets(position, season_range: tuple=(2016, 2024), for_app=False):
    # load datasets
    seasonal_data = pd.read_csv(f"data/quarter_decade/{position}_seasonal_data.csv")
    ngs_data = pd.read_csv(f"data/quarter_decade/{position}_ngs_data.csv")
    roster_data = pd.read_csv(f"data/quarter_decade/{position}_roster_data.csv")

    print("Dataset sizes:")
    print(f"Seasonal: {seasonal_data.shape}")
    print(f"NGS: {ngs_data.shape}")
    print(f"Roster: {roster_data.shape}")

    # standardize player_gsis_id to player_id
    ngs_data = ngs_data.rename(columns={"player_gsis_id": "player_id"})

    # merge the datasets
    comprehensive_data = seasonal_data.merge(
        roster_data,
        on=["player_id", "season"],
        how="inner"
    )

    # TESTING TEAM STATS FOR THE NEXT YEAR'S TEAM FROM LAST YEAR'S STATS

    # merge in team stats
    # team_stats = pd.read_csv("data/team_stats/team_stats_2003_2023.csv")
    # team_stats = team_stats.rename(columns={"year": "season"})
    # team_stats["team"] = team_stats["team"].map(TEAM_NAME_TO_ABBREV)
    # comprehensive_data = comprehensive_data.merge(
    #     team_stats,
    #     on=["team", "season"],
    #     how="inner"
    # )

    # .merge(
    #     ngs_data,
    #     on=["player_id", "season"],
    #     how="left"
    # )

    # optionally filter by seasons
    comprehensive_data = comprehensive_data[
        (comprehensive_data["season"] >= season_range[0]) &
        (comprehensive_data["season"] <= season_range[1])
    ]

    # filter to desired metrics/features
    desired_features = None
    if for_app:
        with open(f"data/metrics/{position}_metrics_app.json", "r") as f:
            desired_features = json.load(f)
    else:
        with open(f"data/metrics/{position}_metrics.json", "r") as f:
            desired_features = json.load(f)

    extra_columns = [
        "player_id",
        "player_name",
        "season",
        "team"
    ]

    columns_to_keep = desired_features + extra_columns
    print(f"{len(columns_to_keep)} being kept")
    filtered_data = comprehensive_data[columns_to_keep]

    print(f"filtered shape: {filtered_data.shape}")
    print(f"unique {position}s: {filtered_data['player_id'].nunique()}")
    print(f"year range: {filtered_data['season'].min()}-{filtered_data['season'].max()}")
    print(f"columns: {len(filtered_data.columns)}")

    # save the new filtered dataset
    filtered_data.to_csv(f"data/filtered/comprehensive_{position}_data.csv", index=False)
    print("Dataset saved")


if __name__ == "__main__":
    merge_datasets("RB")
