import pandas as pd
import json


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
    # .merge(
    #     ngs_data,
    #     on=["player_id", "season"],
    #     how="left"
    # )

    # print("Before NGS merge:")
    # print(f"Unique players: {seasonal_data.merge(roster_data, on=['player_id', 'season'], how='inner')['player_id'].nunique()}")
    # print(f"Seasons covered: {seasonal_data.merge(roster_data, on=['player_id', 'season'], how='inner')['season'].unique()}")

    # print("\nAfter NGS merge:")
    # print(f"Unique players: {comprehensive_data['player_id'].nunique()}")
    # print(f"Seasons covered: {comprehensive_data['season'].unique()}")

    # # Check which seasons have NGS data
    # print(f"\nNGS data seasons: {ngs_data['season'].unique()}")
    # print(f"NGS data player count by season: {ngs_data.groupby('season')['player_id'].nunique()}")

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
        "season"
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
    merge_datasets("WR")
