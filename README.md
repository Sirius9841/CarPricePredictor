# CarPricePredictor

A machine learning pipeline that estimates used car prices using CatBoost. I built this to practice putting together a full ML project the way it'd actually be done in production, not just a single notebook — separate modules for data, preprocessing, feature engineering, training, evaluation, and inference.

Give it details about a car (make, model, year, mileage, condition, accident history, etc.) and it returns a price estimate with a confidence range:

```
Predicted Price:      EUR      18,450.00
Confidence Range:     EUR 15,682.50 - EUR 21,217.50
```

## About the data

I couldn't find a real used-car dataset that was clean and detailed enough for what I wanted to build (consistent fields, enough categorical detail like trim/service history/accident status), so I wrote a synthetic data generator (`data_generator.py`) that simulates the German used-car market — depreciation curves by age and mileage, brand and model pricing tiers, luxury brand effects, and randomized noise so it isn't unrealistically clean.

I did try training on a real dataset too, but the accuracy was noticeably lower, which makes sense — real listings have pricing noise from things like seller motivation and regional demand that a simulator won't capture. So the ~7-8% MAPE below is the model learning my simulator's pricing logic well, not a claim about real-world accuracy. The part I actually wanted to demonstrate is the pipeline itself.

## Project structure

- `data_generator.py` — simulates used car listings with realistic pricing logic
- `preprocessing.py` — cleaning, missing values, outlier clipping
- `feature_engineering.py` — age/mileage interactions, brand tiers, market multipliers
- `train.py` — CatBoost training with early stopping
- `evaluate.py` — cross-validation, residual analysis, error breakdown by segment
- `predict.py` — loads the saved model, includes an interactive CLI
- `config.py` — config as dataclasses instead of scattered constants
- `utils.py` — logging, metrics, shared helpers

## Results

Latest run, 10,000 synthetic listings, CatBoost with early stopping (best iteration 1975/2000):

| Metric | Value |
|---|---|
| MAE | €1,678 |
| RMSE | €3,190 |
| R² | 0.981 |
| MAPE | 7.65% |

Top features by importance: `luxury_brand`, `make`, and `model` dominate, which tracks — brand is the single biggest price driver in the used car market, synthetic or real. `age_mileage_interaction` and the age-based features pick up most of what's left.

## Stack

CatBoost, pandas, NumPy, scikit-learn, joblib.

## Running it

Run these from the project root, not from inside `src/` (the paths and imports assume that):

```bash
python src/data_generator.py --samples 10000
python src/train.py
python src/predict.py
```

## Next steps

Testing against a bigger real-world dataset properly, tuning hyperparameters with Optuna (already scaffolded in the config), and wrapping the model in a small API.
