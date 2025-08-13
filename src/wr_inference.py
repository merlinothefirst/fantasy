import pandas as pd
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning
import json


MODEL = "linear_regression"
POSITION = "WR"
TOP_N = 50

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# load model and features
print("Loading trained model...")
model = joblib.load(f"models/{MODEL}_wr_model_weights.pkl")

# with open(f"models/{MODEL}_wr_model_stats.json", "r") as f:
#     model_info = json.load(f)
# feature_columns = model_info["feature_columns"]
with open("data/quarter_decade/WR_metrics.json", "r") as f:
    desired_metrics = json.load(f)
desired_metrics.append("fantasy_points_ppr")
feature_columns = desired_metrics

# load data
data = pd.read_csv("data/filtered/comprehensive_WR_data.csv")

# create 2024 season totals (input features)
last_season_totals = data[data["season"] == 2024].copy()
last_season_totals = last_season_totals.drop_duplicates(subset="player_name", keep="first")
print(f"Found {len(last_season_totals)} WRs with 2024 data")

# prepare features for prediction
prediction_feautures = last_season_totals[feature_columns].fillna(last_season_totals[feature_columns].median())

# make predictions for 2025
predictions_2025 = model.predict(prediction_feautures)

# organize results
results = pd.DataFrame({
    "player_id": last_season_totals["player_id"],
    "player_name": last_season_totals["player_name"],
    "predicted_2025_points": predictions_2025
})

# sort and display top N predictions
top_predictions = results.sort_values("predicted_2025_points", ascending=False).head(TOP_N)

print()
print("WR rank:  Player:             Predicted PPR points:")
print("-----------------------------------------")
for ix, row in top_predictions.iterrows():
    print(f"{str(ix+1):10}{row['player_name']:35}{row['predicted_2025_points']:.2f}")

print()
print("Checking feature alignment...")
print(f"Training features: {len(feature_columns)}")
print(f"Available in 2024 data: {sum([col in last_season_totals.columns for col in feature_columns])}")
missing = [col for col in feature_columns if col not in last_season_totals.columns]
if missing:
    print(f"Missing features: {missing}")