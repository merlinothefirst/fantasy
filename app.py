from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import joblib
import os
from pathlib import Path
import sys

# Add your src directory to path
sys.path.append(str(Path(__file__).resolve().parents[0] / 'src'))

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Import your existing merge function
from merge_dataset import merge_datasets

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Available positions and models
POSITIONS = ["QB", "RB", "WR", "TE"]
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, enable_categorical=True),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get available positions"""
    return jsonify(POSITIONS)

@app.route('/api/features/<position>', methods=['GET'])
def get_features(position):
    """Get available features for a position"""
    try:
        with open(f"data/metrics/{position}_metrics_app.json", "r") as f:
            features = json.load(f)
        return jsonify(features)
    except FileNotFoundError:
        return jsonify({"error": f"Features file not found for {position}"}), 404

@app.route('/api/train-and-predict', methods=['POST'])
def train_and_predict():
    """Train models with selected features and return predictions"""
    try:
        data = request.json
        position = data['position']
        selected_features = data['features']
        top_n = data.get('top_n', 20)
        
        print(f"Training {position} with {len(selected_features)} features...")
        
        # Step 1: Merge datasets (reuse your existing logic)
        merge_datasets(position, for_app=True)
        
        # Step 2: Load and prepare data (from your training script)
        comprehensive_data = pd.read_csv(f"data/filtered/comprehensive_{position}_data.csv")
        
        # Create prediction pairs (your existing logic)
        unique_players = comprehensive_data["player_id"].unique()
        prediction_pairs = []

        for player_id in unique_players:
            player_data = comprehensive_data[comprehensive_data["player_id"] == player_id].sort_values("season")

            for i in range(len(player_data) - 1):
                current_row = player_data.iloc[i]
                next_row = player_data.iloc[i + 1]

                current_szn = current_row["season"]
                next_szn = next_row["season"]

                if next_szn == current_szn + 1:
                    features = current_row.drop(["player_id", "player_name", "season"])
                    target = next_row["fantasy_points_ppr"]

                    prediction_pair = {
                        "player_id": player_id,
                        "player_name": current_row["player_name"],
                        "stats_season": current_szn,
                        "predicted_season": next_szn,
                        "target": target,
                        **features.to_dict()
                    }
                    prediction_pairs.append(prediction_pair)

        prediction_pairs_df = pd.DataFrame(prediction_pairs)
        prediction_pairs_df = prediction_pairs_df.sort_values("player_name")
        
        # Use only selected features
        X = prediction_pairs_df[selected_features]
        y = prediction_pairs_df["target"]
        
        # Handle infinite/NaN values
        X = X.fillna(X.median())
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Step 3: Train all models and find best one
        best_model = None
        best_r2 = -float('inf')
        best_model_name = None
        model_results = {}
        
        for name, model in MODELS.items():
            try:
                print(f"Training {name}...")
                
                if name == "XGBoost":
                    X_train_model = X_train
                    X_test_model = X_test
                else:
                    X_train_model = X_train.fillna(X_train.median())
                    X_test_model = X_test.fillna(X_test.median())
                
                model.fit(X_train_model, y_train)
                y_pred = model.predict(X_test_model)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                model_results[name] = {"R2": r2, "RMSE": rmse}
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"{name} training failed: {e}")
                model_results[name] = {"R2": None, "RMSE": None, "error": str(e)}
        
        if best_model is None:
            return jsonify({"error": "All models failed to train"}), 500
            
        print(f"Best model: {best_model_name} (R2: {best_r2:.3f})")
        
        # Step 4: Run inference on 2024 data (your existing logic)
        last_season_data = comprehensive_data[comprehensive_data["season"] == 2024].copy()
        last_season_data = last_season_data.drop_duplicates(subset="player_name", keep="first")
        
        if len(last_season_data) == 0:
            return jsonify({"error": "No 2024 data found for predictions"}), 400
        
        # Prepare features for prediction
        prediction_features = last_season_data[selected_features].fillna(last_season_data[selected_features].median())
        
        # Make predictions
        predictions_2025 = best_model.predict(prediction_features)
        
        # Organize results
        results = pd.DataFrame({
            "player_id": last_season_data["player_id"],
            "player_name": last_season_data["player_name"],
            "predicted_2025_points": predictions_2025
        })
        
        # Get top N predictions
        top_predictions = results.sort_values("predicted_2025_points", ascending=False).head(top_n)
        
        # Format response
        response = {
            "position": position,
            "features_used": selected_features,
            "best_model": {
                "name": best_model_name,
                "r2_score": best_r2,
                "rmse": model_results[best_model_name]["RMSE"]
            },
            "all_model_results": model_results,
            "predictions": [
                {
                    "rank": idx + 1,
                    "player_name": row["player_name"],
                    "predicted_points": round(row["predicted_2025_points"], 2)
                }
                for idx, (_, row) in enumerate(top_predictions.iterrows())
            ],
            "total_players": len(results)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in train_and_predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)