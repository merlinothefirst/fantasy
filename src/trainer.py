import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.merge_dataset import merge_datasets


def run_training(position: str):
    pos = position.lower()
    print(f"-------------------- {position}s --------------------")

    # first need to create prediction pairs: X matrix's vectors correspond to y vector's values,
    # where the X matrix vector is metrics from a given year, and y vector's value is the fantasy
    # output of the year after

    comprehensive_data = pd.read_csv(f"data/filtered/comprehensive_{position}_data.csv")

    unique_players = comprehensive_data["player_id"].unique()
    print(f"{len(unique_players)} unique {position}s found")

    prediction_pairs = []

    for player_id in unique_players:
        player_data = comprehensive_data[comprehensive_data["player_id"] == player_id].sort_values("season")

        for i in range(len(player_data) - 1):
            current_row = player_data.iloc[i]
            next_row = player_data.iloc[i + 1]

            current_szn = current_row["season"]
            next_szn = next_row["season"]

            if next_szn == current_szn + 1:
                # X vector for given pair:
                features = current_row.drop(["player_id", "player_name", "season"])

                # y value for give pair:
                target = next_row["fantasy_points_ppr"]

                prediction_pair = {
                    # save metadata, aka year and player info
                    "player_id": player_id,
                    "player_name": current_row["player_name"],
                    "stats_season": current_szn,
                    "predicted_season": next_szn,

                    # y value
                    "target": target,

                    # X vector
                    **features.to_dict()
                }

                prediction_pairs.append(prediction_pair)

    prediction_pairs_df = pd.DataFrame(prediction_pairs)
    prediction_pairs_df = prediction_pairs_df.sort_values("player_name")

    print(prediction_pairs_df.head())
    print(prediction_pairs_df.shape)

    # next we need to convert this into the appropriate X matrix and y vector for learning
    with open(f"data/metrics/{position}_metrics.json", "r") as f:
        desired_metrics = json.load(f)

    X = prediction_pairs_df[desired_metrics]
    y = prediction_pairs_df["target"]

    mask_inf = ~np.isfinite(X)  # True for inf, -inf, or NaN
    bad_locations = np.where(mask_inf)

    # Print the offending rows & columns
    for row, col in zip(*bad_locations):
        print(f"Row {row}, Column '{X.columns[col]}', Value: {X.iat[row, col]}")

    X = X.fillna(X.median())

    # select k-best features
    selector_f = SelectKBest(score_func=f_regression, k=10)
    selector_f.fit_transform(X, y)

    feature_scores = selector_f.scores_
    feature_names = X.columns
    feature_ranking = sorted(zip(feature_names, feature_scores), key=lambda x: x[1], reverse=True)

    print("Feature importance (F-statistic):")
    for feature, score in feature_ranking:
        selected = f"!" if feature in X.columns[selector_f.get_support()] else "  "
        print(f"{selected} {feature:25} {score:8.2f}")
    print()

    # create splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # correlation
    correlation_matrix = X_train.corr()
    high_corr = correlation_matrix[correlation_matrix.abs() > 0.8]
    print("Highly correlated features:")
    print(high_corr)

    # define a variety of models for testing
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, enable_categorical=True),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1)
    }

    # iterate through and test each model
    results = {}

    for name, model in models.items():
        print(f"-------------------- {name} --------------------")

        X_train_model = X_train
        X_test_model = X_test

        try:
            # train the actual current model
            model.fit(X_train_model, y_train)

            # make predictions to evaluate performance
            y_pred = model.predict(X_test_model)

            # calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {"R2": r2, "RMSE": rmse}
            print(f"{name} completed training: R2 = {r2:.3f}, RMSE = {rmse:.1f}")

            joblib.dump(model, f"models/{name.replace(' ', '_').lower()}_{pos}_model_weights.pkl")
            model_info = {
                "feature_columns": list(X_train.columns),
                "model_performance": {"R2": r2, "RMSE": rmse},
                "training_samples": len(X_train),
                "nan_handling": "fill_nan_with_median"
            }
            with open(f"models/{name.replace(' ', '_').lower()}_{pos}_model_stats.json", "w") as f:
                json.dump(model_info, f, indent=2)

        except Exception as e:
            print(f"{name} training failed: {e}")
            results[name] = {"R2": None, "RMSE": None}


if __name__ == "__main__":
    POS = "WR"

    merge_datasets(POS)

    run_training(POS)
