# Electricity Consumption Forecasting — Bukit Mahkota, Bangi

LSTM model predicting hourly residential electricity consumption from historical
usage + weather data, served through a Flask dashboard + interactive forecast tool.

## Environment

System Python here is the Windows Store install, whose site-packages path is too
long for some packages (TensorFlow) to install under Windows's 260-character path
limit. A dedicated virtual environment lives at:

```
C:\Users\mukmi\pyenvs\lstm
```

Use its interpreter for everything in this project:

```
C:\Users\mukmi\pyenvs\lstm\Scripts\python.exe -m pip install -r requirements.txt
```

A Jupyter kernel for it is already registered as **"Python (lstm-electricity)"**.

## Project layout

```
data/                    raw electricitysupp.csv + weather.xlsx, and data/processed/ (generated)
src/
  config.py              all paths & hyperparameters
  data_cleaning.py        raw -> cleaned, merged hourly DataFrame
  feature_engineering.py  time/lag/rolling features
  dataset.py               chronological split, scaling, LSTM sequence windowing
  model.py                  LSTM architecture
  train.py / evaluate.py    training and test-set evaluation scripts
  forecast.py                recursive multi-hour forecasting (used by the web app)
  importance.py               permutation feature importance (used by the dashboard)
notebooks/                end-to-end walkthrough notebook
models/, outputs/         generated: trained model, scalers, metrics, plots
app.py                    Flask web app (dashboard + interactive predictor)
templates/, static/        Flask app's HTML/CSS/JS
```

## Retrain from scratch

```
C:\Users\mukmi\pyenvs\lstm\Scripts\python.exe -m src.train
C:\Users\mukmi\pyenvs\lstm\Scripts\python.exe -m src.evaluate
C:\Users\mukmi\pyenvs\lstm\Scripts\python.exe -m src.importance   # for the dashboard's "main factor" panel
```

## Run the web app

```
C:\Users\mukmi\pyenvs\lstm\Scripts\python.exe app.py
```

Then open http://127.0.0.1:5000 — Dashboard has the EDA/correlation/model-performance
breakdown, Predict lets you pick any start hour + a 1–72h horizon and forecasts
recursively (each hour's prediction feeds the next hour's lag features). Since there's
no live weather feed, valid start/horizon combinations stay inside the historical
period, which doubles as a backtest: you get an actual-vs-predicted comparison for
free wherever ground truth exists.
