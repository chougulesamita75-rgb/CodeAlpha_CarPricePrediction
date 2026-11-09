# Used Car Price Prediction — Complete Project

## Dataset used
This project includes the exact CSV supplied for this task:
- **301 rows**
- **9 columns**
- Features: `Car_Name`, `Year`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`
- Target: `Selling_Price`

The project derives `Car_Age` from `Year` using the newest year in the dataset as its reference year. `Car_Name` is excluded from model features.

## Folder structure
```text
Car_Price_Prediction_Using_Your_Dataset/
├── dataset/
│   └── car data.csv
├── outputs/
│   ├── car_price_model.joblib
│   ├── model_metrics.csv
│   ├── prediction_results.csv
│   ├── actual_vs_predicted.png
│   └── selling_price_distribution.png
├── car_price_prediction.py
├── predict_car.py
├── requirements.txt
└── README.md
```

## Step 1 — Open in VS Code
Extract the ZIP file and open the extracted `Car_Price_Prediction_Using_Your_Dataset` folder in VS Code.

## Step 2 — Install packages
Open **Terminal → New Terminal** and run:
```bash
python -m pip install -r requirements.txt
```
If `python` does not point to the Python installation you selected in VS Code, use:
```bash
py -m pip install -r requirements.txt
```

## Step 3 — Train and evaluate
```bash
python car_price_prediction.py
```
This performs data inspection, cleaning, preprocessing, an 80/20 train-test split, Random Forest Regression, evaluation, visualizations, and saves a final model trained on all cleaned records.

## Step 4 — Predict a new car price
After training:
```bash
python predict_car.py
```
Enter the requested car details when prompted. Prices are in the same units as the source dataset.

## Evaluation metrics
- **MAE:** average absolute prediction error.
- **RMSE:** square root of average squared prediction error; larger errors receive more weight.
- **R²:** proportion of target-price variation explained by the model.

The metrics in `outputs/model_metrics.csv` are from the held-out test split. They can vary if the dataset or model settings change.

## Important
This is an educational machine-learning model. Its predictions are estimates based on this small dataset, not guaranteed market valuations.
