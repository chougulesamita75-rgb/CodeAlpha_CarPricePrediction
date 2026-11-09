"""Predict a used car's selling price using the trained model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "outputs" / "car_price_model.joblib"

def main():
    if not MODEL_FILE.exists():
        print("Model not found. Run: python car_price_prediction.py")
        return

    model = joblib.load(MODEL_FILE)
    print("\\nCAR PRICE PREDICTION")
    print("Enter the car details below.")

    try:
        present_price = float(input("Present price (in dataset units): "))
        driven_kms = float(input("Kilometres driven: "))
        fuel_type = input("Fuel type (Petrol/Diesel/CNG): ").strip().title()
        selling_type = input("Seller type (Dealer/Individual): ").strip().title()
        transmission = input("Transmission (Manual/Automatic): ").strip().title()
        owner = int(input("Previous owners (0/1/2/3): "))
        car_age = int(input("Car age in years: "))

        sample = pd.DataFrame([{
            "Present_Price": present_price,
            "Driven_kms": driven_kms,
            "Fuel_Type": fuel_type,
            "Selling_type": selling_type,
            "Transmission": transmission,
            "Owner": owner,
            "Car_Age": car_age
        }])
        price = model.predict(sample)[0]
        print(f"\\nEstimated selling price: {price:.2f} (same units as the dataset)")
    except ValueError:
        print("Invalid input. Please enter numbers where requested.")

if __name__ == "__main__":
    main()
