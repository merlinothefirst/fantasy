import nfl_data_py as nfl
import json

# functions = [func for func in dir(nfl) if not func.startswith('_')]
# for func in functions:
#     print(func)
# print()

# weekly_columns = nfl.see_weekly_cols()
# print("Weekly data columns:")
# print(weekly_columns)
# print()

fns = [fn for fn in dir(nfl) if not fn.startswith("_")]
print(fns)

# player_data = nfl.import_players()
# methods = [method for method in dir(player_data) if not method.startswith("_")]
# print()
# print(methods)
# print()
# print(player_data.head())
# player_data.info()

season_data = nfl.import_seasonal_data([2024], "REG")
# methods = [method for method in dir(season_data) if not method.startswith("_")]

# print(season_data.head())
season_data.info()

ngs_data = nfl.import_ngs_data("rushing", [2024])
ngs_data.info()

roster_data = nfl.import_seasonal_rosters([2024])
roster_data.info()

wr_metrics = []
for ix, column in enumerate(season_data.columns):
    if (ix >= 19 and ix <= 41) or (ix >= 44 and ix <= 57):
        wr_metrics.append(column)

for ix, column in enumerate(ngs_data.columns):
    if (ix >= 6 and ix <= 12) or (ix >= 18):
        wr_metrics.append(column)

for ix, column in enumerate(roster_data.columns):
    if (ix == 36 or ix == 35 or ix == 10 or ix == 11 or ix == 22):
        wr_metrics.append(column)

with open("data/metrics/RB_metrics.json", "w") as f:
    json.dump(wr_metrics, f, indent=2)

print("Finished")

# weekly_data = nfl.import_weekly_data([2024])
# print("Available methods on weekly_data DataFrame:")
# methods = [method for method in dir(weekly_data) if not method.startswith('_')]
# for method in methods:
#     print(method)

# spec_data = weekly_data[weekly_data["player_display_name"] == "Tyreek Hill"]
# print(spec_data["rushing_yards"])

# weekly_data = nfl.import_weekly_data([2024])
# pbp_data = nfl.import_pbp_data([2024])


# # Calculate pass rate by team/season
# team_pass_rates = pbp_data.groupby(['season', 'posteam']).agg({
#     'pass': 'mean'  # This gives us pass rate
# }).reset_index()

# # Filter for WRs only before calculating defense stats
# wr_data = weekly_data[weekly_data['position'] == 'WR']

# # For opponent WR points allowed - use WR data only
# defense_vs_wr = wr_data.groupby(['season', 'opponent_team', 'week']).agg({
#     'fantasy_points_ppr': 'sum'  # Total points per game
# }).reset_index().groupby(['season', 'opponent_team']).agg({
#     'fantasy_points_ppr': 'mean'  # Average total points per game
# }).reset_index()

# # Let's see what we got
# # Most pass-heavy teams
# print("Most pass-heavy teams:")
# print(team_pass_rates.sort_values('pass', ascending=False).head(5))

# # Most run-heavy teams
# print("\nMost run-heavy teams:")
# print(team_pass_rates.sort_values('pass', ascending=True).head(5))

# # Worst defenses vs WRs (allow most points)
# print("\nWorst defenses vs WRs:")
# print(defense_vs_wr.sort_values('fantasy_points_ppr', ascending=False).head(5))

# # Best defenses vs WRs (allow fewest points)
# print("\nBest defenses vs WRs:")
# print(defense_vs_wr.sort_values('fantasy_points_ppr', ascending=True).head(5))
