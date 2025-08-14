import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("AUTOMATED FEATURE SELECTION FOR WR PREDICTION")
print("=" * 50)

# Load your existing training data (from your training script)
# step 1: load and combine WR data
seasons = range(2000, 2025)
all_wr_data = []

for season in seasons:
    wr_data = pd.read_csv(f"data/raw/WR_{season}_weekly.csv")
    all_wr_data.append(wr_data)

combined_wr_data = pd.concat(all_wr_data, ignore_index=True)
print(f"Combined data: {len(combined_wr_data)} total WR player-weeks")

# step 2: determine season totals for each player-season
seasonal_stats = combined_wr_data.groupby(["player_display_name", "season"]).agg({
    "fantasy_points_ppr": "sum",  # y vector
    # Core stats
    "targets": "sum",
    "receptions": "sum", 
    "receiving_yards": "sum",
    "receiving_tds": "sum",
    "receiving_fumbles": "sum",
    # Advanced metrics (use mean for rates/percentages)
    "target_share": "mean",
    "air_yards_share": "mean", 
    "receiving_air_yards": "sum",
    "receiving_yards_after_catch": "sum",

    "receiving_epa": "sum",
    "receiving_first_downs": "sum",
    # Advanced efficiency metrics
    "wopr": "mean",
    "racr": "mean"
}).reset_index()

print(f"{len(seasonal_stats)} player-seasons created")
print(seasonal_stats.head())
print()

# step 3: use a previous season's stats to predict next season's total points, and accumulate neatly
prediction_data = []
for player in seasonal_stats["player_display_name"].unique():
    player_data = seasonal_stats[seasonal_stats["player_display_name"] == player].sort_values("season")

    for i in range(len(player_data) - 1):
        current_szn = player_data.iloc[i]
        next_szn = player_data.iloc[i + 1]

        if next_szn["season"] != current_szn["season"] + 1:
            continue

        prediction_pair = {
            "player": player,
            "predict_season": next_szn["season"],
            # All the features - ADD THE MISSING ONES:
            "prev_targets": current_szn["targets"],
            "prev_receptions": current_szn["receptions"],
            "prev_receiving_yards": current_szn["receiving_yards"],
            "prev_receiving_tds": current_szn["receiving_tds"],
            "prev_receiving_fumbles": current_szn["receiving_fumbles"],

            "prev_target_share": current_szn["target_share"],
            "prev_air_yards_share": current_szn["air_yards_share"],
            "prev_receiving_air_yards": current_szn["receiving_air_yards"],
            "prev_receiving_yards_after_catch": current_szn["receiving_yards_after_catch"],

            "prev_receiving_epa": current_szn["receiving_epa"],
            "prev_receiving_first_downs": current_szn["receiving_first_downs"],
            "prev_wopr": current_szn["wopr"],
            "prev_racr": current_szn["racr"],

            "next_fantasy_points": next_szn["fantasy_points_ppr"]
        }
        prediction_data.append(prediction_pair)

prediction_df = pd.DataFrame(prediction_data)
print(f"Created {len(prediction_df)} prediction pairs")

prediction_df_clean = prediction_df.dropna()

print(prediction_df_clean.head(10))

# step 4: create X matrix and y vector
feature_columns = ["prev_targets", "prev_target_share", "prev_receiving_yards", "prev_receptions", "prev_receiving_tds", "prev_air_yards_share"]
X = prediction_df_clean[feature_columns]
y = prediction_df_clean["next_fantasy_points"]

print(f"X matrix shape: {X.shape}")
print(f"y vector shape: {y.shape}")

# Step 1: Create a comprehensive feature set for WRs
wr_features = [
   # Core volume stats
   'prev_targets', 
   'prev_receptions', 
   'prev_receiving_yards',
   'prev_receiving_tds',
   'prev_receiving_fumbles',
   
   # Advanced opportunity metrics
   'prev_target_share',
   'prev_air_yards_share', 
   'prev_receiving_air_yards',
   'prev_receiving_yards_after_catch',
   
   # Efficiency and advanced metrics
   'prev_wopr',
   'prev_racr',
   'prev_receiving_epa',
   'prev_receiving_first_downs',
]

# Make sure we only use features that exist in your data
available_features = [f for f in wr_features if f in prediction_df_clean.columns]
print(f"Available WR features: {len(available_features)}")
for feature in available_features:
    print(f"  - {feature}")

X_full = prediction_df_clean[available_features]
y = prediction_df_clean['next_fantasy_points']

print(f"\nFull feature matrix: {X_full.shape}")

# Method 1: SelectKBest with F-statistics
print("\n1. SELECTKBEST - Top features by F-statistic")
selector_f = SelectKBest(score_func=f_regression, k=4)
X_selected_f = selector_f.fit_transform(X_full, y)

# Get feature scores
feature_scores = selector_f.scores_
feature_names = X_full.columns
feature_ranking = sorted(zip(feature_names, feature_scores), key=lambda x: x[1], reverse=True)

print("Feature importance (F-statistic):")
for feature, score in feature_ranking:
    selected = "✅" if feature in X_full.columns[selector_f.get_support()] else "  "
    print(f"{selected} {feature:25} {score:8.2f}")

selected_features_f = X_full.columns[selector_f.get_support()].tolist()
print(f"\nTop 4 features: {selected_features_f}")

# Test different feature sets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Compare 4 vs 8 vs 13 features
feature_sets = {
    'top_4': selected_features_f,  # Your top 4
    'top_8': [f[0] for f in feature_ranking[:8]],  # Top 8
    'all_13': available_features   # All features
}

for name, features in feature_sets.items():
    X_temp = prediction_df_clean[features]
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    print(f"{name:6} ({len(features):2d} features): R² = {r2:.3f}")