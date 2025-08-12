import nfl_data_py as nfl


print("SAVING DATA...")

seasons = range(2000, 2025)
positions = ["WR", "RB", "TE", "QB"]

for season in seasons:
    print(f"Downloading season {season}...")
    season_data = nfl.import_weekly_data([season])

    for position in positions:
        pos_season_data = season_data[season_data["position"] == position].copy()

        num_players = pos_season_data["player_display_name"].nunique()
        print(f"{num_players} unique {position}s in {season}")
    
        filename = f"data/raw/{position}_{season}_weekly.csv"
        pos_season_data.to_csv(filename, index=False)
        print(f"Saved to {filename}")

print("Raw data saved")