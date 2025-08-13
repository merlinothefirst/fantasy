import pandas as pd
import json


# load datasets
seasonal_data = pd.read_csv("data/quarter_decade/WR_seasonal_data.csv")
ngs_data = pd.read_csv("data/quarter_decade/WR_ngs_data.csv")
roster_data = pd.read_csv("data/quarter_decade/WR_roster_data.csv")

print("Dataset sizes:")
print(f"Seasonal: {seasonal_data.shape}")
print(f"NGS: {ngs_data.shape}")  
print(f"Roster: {roster_data.shape}")

# standardize player_gsis_id to player_id
ngs_data = ngs_data.rename(columns={"player_gsis_id": "player_id"})

# merge the datasets
comprehensive_data = seasonal_data.merge(
    ngs_data,
    on=["player_id", "season"],
    how="inner"
).merge(
    roster_data,
    on=["player_id", "season"],
    how="inner"
)

# filter to desired metrics/features
with open("data/quarter_decade/WR_metrics.json", "r") as f:
    desired_features = json.load(f)

extra_columns = [
    "player_id",
    "player_name",
    "season",
    "fantasy_points_ppr"
]

columns_to_keep = desired_features + extra_columns
print(f"{len(columns_to_keep)} being kept")
filtered_data = comprehensive_data[columns_to_keep]

print(f"filtered shape: {filtered_data.shape}")
print(f"unique WRs: {filtered_data['player_id'].nunique()}")
print(f"year range: {filtered_data['season'].min()}-{filtered_data['season'].max()}")
print(f"columns: {len(filtered_data.columns)}")

# save the new filtered dataset
filtered_data.to_csv("data/filtered/comprehensive_WR_data.csv", index=False)
print("Dataset saved")