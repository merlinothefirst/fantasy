import pandas as pd
import joblib
import warnings
from sklearn.exceptions import DataConversionWarning


MODEL_PATH = "models/wr_model_weights.pkl"
POSITION = "WR"
TOP_N = 50

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# step 1: load model
print("Loading trained model...")
model = joblib.load(MODEL_PATH)

# step 2: load data
data = pd.read_csv(f"data/raw/{POSITION}_2024_weekly.csv")

# step 3: create 2024 season totals (input features)
last_szn_totals = data.groupby("player_display_name").agg({
    "targets": "sum",
    "target_share": "mean",
    "receiving_yards": "sum",
    "receptions": "sum",
    "receiving_tds": "sum",
    "air_yards_share": "mean",
}).reset_index()

print(f"{len(last_szn_totals)} players available for prediction")

# step 4: determine players for prediction
# players = ["CeeDee Lamb", "Ja\'Marr Chase", "Justin Jefferson"]
input_players = pd.read_csv("data/inference_input/input_players.csv", skiprows=4)
players = []
for ix, row in input_players.iterrows():
    if ix > TOP_N:
        break

    player = row["Player"]

    # inputs don't have Jr.'s:
    if player.endswith("Jr."):
        player = player[:-4]

    # DJ Moore:
    if "DJ" in player:
        player = player.replace("DJ", "D.J.")

    if row["Pos"] == POSITION:
        players.append(player)

print(f"Players to predict: {players}")
print()

# step 5: run prediction
predictions_2025 = []
for player in players:
    player_2024 = last_szn_totals[last_szn_totals["player_display_name"] == player]

    if player_2024.empty:
        print(f"Couldn't find {player}")
        continue

    features = [[
        player_2024["targets"].iloc[0],
        player_2024["target_share"].iloc[0],
        player_2024["receiving_yards"].iloc[0],
        player_2024["receptions"].iloc[0],
        player_2024["receiving_tds"].iloc[0],
        player_2024["air_yards_share"].iloc[0]
    ]]

    predicted_2025 = model.predict(features)[0]

    predictions_2025.append({
        "player": player,
        "predicted_2025": predicted_2025
    })

# step 6: return results
predictions_2025.sort(key=lambda x: x["predicted_2025"], reverse=True)
print("WR rank:  Player:             Predicted PPR points:")
print("-----------------------------------------")
for ix, prediction in enumerate(predictions_2025):
    print(f"{str(ix+1):10}{prediction['player']:35}{prediction['predicted_2025']:.2f}")
