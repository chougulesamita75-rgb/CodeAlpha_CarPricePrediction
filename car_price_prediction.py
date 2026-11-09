"""
USED CAR PRICE PREDICTION
Dataset: dataset/car data.csv
Run: python car_price_prediction.py
"""

from pathlib import Path
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "dataset" / "car data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    print("\\n" + "=" * 58)
    print("             USED CAR PRICE PREDICTION")
    print("=" * 58)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    print("\\nDataset shape:", data.shape)
    print("\\nColumns:", list(data.columns))
    print("\\nFirst five records:\\n", data.head().to_string(index=False))
    print("\\nMissing values:\\n", data.isnull().sum().to_string())

    # Clean data.
    data.columns = data.columns.str.strip()
    data = data.drop_duplicates().copy()
    data["Selling_Price"] = pd.to_numeric(data["Selling_Price"], errors="coerce")
    data = data.dropna(subset=["Selling_Price"])

    # Create car age from the dataset's Year column.
    # Reference year is set to the newest year in this dataset for reproducibility.
    reference_year = int(data["Year"].max())
    data["Car_Age"] = reference_year - pd.to_numeric(data["Year"], errors="coerce")
    data = data.drop(columns=["Year", "Car_Name"], errors="ignore")

    X = data.drop(columns=["Selling_Price"])
    y = data["Selling_Price"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, min_samples_leaf=1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\\nMODEL EVALUATION")
    print("-" * 38)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Save test predictions and metrics.
    pd.DataFrame({
        "Actual Selling Price": y_test.to_numpy(),
        "Predicted Selling Price": predictions
    }).to_csv(OUT / "prediction_results.csv", index=False)

    pd.DataFrame([{
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Training Rows": len(X_train), "Testing Rows": len(X_test)
    }]).to_csv(OUT / "model_metrics.csv", index=False)

    # Actual versus predicted chart.
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, predictions, alpha=0.75)
    lower = min(float(y_test.min()), float(predictions.min()))
    upper = max(float(y_test.max()), float(predictions.max()))
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title("Actual vs Predicted Car Prices")
    plt.tight_layout()
    plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Dataset visualizations.
    plt.figure(figsize=(8, 5))
    data["Selling_Price"].hist(bins=20)
    plt.xlabel("Selling Price")
    plt.ylabel("Number of Cars")
    plt.title("Distribution of Selling Prices")
    plt.tight_layout()
    plt.savefig(OUT / "selling_price_distribution.png", dpi=160)
    plt.close()

    # Retrain on all cleaned records for the final model used for new predictions.
    final_model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, min_samples_leaf=1
        ))
    ])
    final_model.fit(X, y)
    joblib.dump(final_model, OUT / "car_price_model.joblib")

    print("\\nSaved files in outputs/:")
    print("- car_price_model.joblib")
    print("- model_metrics.csv")
    print("- prediction_results.csv")
    print("- actual_vs_predicted.png")
    print("- selling_price_distribution.png")
    print("\\nProject completed successfully!")

if __name__ == "__main__":
    main()
