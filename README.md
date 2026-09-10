# Used Car Price Predictor

A regression project that estimates used-car prices from vehicle specifications, history, and appearance. The notebook includes substantial text cleanup and feature engineering before comparing linear and tree-based regression models.

## Workflow

- Parse currency and mileage strings and clean missing or placeholder values.
- Explore price, mileage, model year, accident history, and brand-level patterns.
- Correct multiword brand names and extract horsepower, engine size, and cylinder count.
- Identify electric vehicles and simplify transmission descriptions.
- Group rare models and colors to control categorical dimensionality.
- Encode categorical fields and standardize features for linear regression.
- Compare linear regression and random forest on raw and log-transformed targets.
- Evaluate MAE, RMSE, and R-squared, including a lower-price subset.
- Tune random-forest hyperparameters and inspect feature importance.
- Export a random-forest model and standard scaler.

## Dataset

`used_cars.csv` contains 4,009 listings. Fields include brand, model, model year, mileage, fuel type, engine description, transmission, exterior and interior colors, accident history, title status, and price.

## Repository contents

| File | Purpose |
| --- | --- |
| `Used-Cars-Price-Predictor.ipynb` | Cleaning, feature engineering, model comparison, tuning, and export |
| `used_cars.csv` | Source listing dataset |
| `used_car_price_model.joblib` | Exported random-forest regressor |
| `used_car_price_scaler.joblib` | Exported standard scaler |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn joblib
jupyter lab Used-Cars-Price-Predictor.ipynb
```

Inference requires reproducing the notebook's feature-engineering and one-hot-encoding columns. Note that the exported model is the baseline random forest assigned to `rf`, while the notebook also evaluates a separately tuned estimator.
