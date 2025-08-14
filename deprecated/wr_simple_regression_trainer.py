import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from xgboost import XGBRegressor

print("Let's build a WR prediction regression model")

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

    "receiving_yards": "sum",
    "receiving_first_downs": "sum",
    "receptions": "sum",
    "targets": "sum",
    "target_share": "mean",
    "receiving_yards_after_catch": "sum",
    "wopr": "mean",
    "receiving_air_yards": "sum"
    # "receiving_tds": "sum",
    # "air_yards_share": "mean"
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

            # features (X): from current season
            "prev_receiving_yards": current_szn["receiving_yards"],
            "prev_receiving_first_downs": current_szn["receiving_first_downs"],
            "prev_receptions": current_szn["receptions"],
            "prev_targets": current_szn["targets"],
            "prev_target_share": current_szn["target_share"],
            "prev_receiving_yards_after_catch": current_szn["receiving_yards_after_catch"],
            "prev_wopr": current_szn["wopr"],
            "prev_receiving_air_yards": current_szn["receiving_air_yards"],
            # "prev_receiving_tds": current_szn["receiving_tds"],
            # "prev_air_yards_share": current_szn["air_yards_share"],

            # target (y): from next season
            "next_fantasy_points": next_szn["fantasy_points_ppr"]
        }
        prediction_data.append(prediction_pair)

prediction_df = pd.DataFrame(prediction_data)
print(f"Created {len(prediction_df)} prediction pairs")

prediction_df_clean = prediction_df.dropna()

print(prediction_df_clean.head(10))
print(prediction_df_clean.shape)

# step 4: create X matrix and y vector
feature_columns = [
    "prev_receiving_yards",
    "prev_receiving_first_downs",
    "prev_receptions",
    "prev_targets",
    "prev_target_share",
    "prev_receiving_yards_after_catch",
    "prev_wopr",
    "prev_receiving_air_yards",
]
X = prediction_df_clean[feature_columns]
y = prediction_df_clean["next_fantasy_points"]

print(f"X matrix shape: {X.shape}")
print(f"y vector shape: {y.shape}")

# step 5: split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# step 6: train the model
print("Training via Linear Regression...")
print()
model = LinearRegression()
# model = XGBRegressor()
model.fit(X_train, y_train)

# step 7: test the model
y_pred = model.predict(X_test)

# step 8: evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print("MODEL RESULTS:")
print(f"R^2 score: {r2:.3f}")
print(f"RMSE: {rmse:.1f} fantasy points")

# step 9: save training weight
print("Saving trained model...")
joblib.dump(model, "models/wr_model_weights.pkl")

feature_info = {
    "feature_columns": feature_columns,
    "model_metrics": {
        "r2_score": r2,
        "rmse": rmse,
        "n_train_samples": len(X_train)
    }
}
with open("models/wr_model_stats.json", "w") as f:
    json.dump(feature_info, f, indent=2)

print("Model weights and stats successfully saved")
