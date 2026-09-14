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
app.py                    Flask web app (dashboard + interactive predictor) — for local use
templates/, static/        Flask app's HTML/CSS/JS
docs/                     static, client-side rebuild of the same app (TensorFlow.js) — for GitHub Pages
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

## Deploy to GitHub Pages (docs/)

GitHub Pages only serves static files — it can't run Flask or Python. `docs/` is a
full rebuild of the same dashboard + predictor that runs **entirely in the browser**
via TensorFlow.js: the trained model's weights and the fitted scalers are exported to
plain JSON (`src/export_web.py`), a small hand-written loader reconstructs the exact
same Keras architecture in `tf.layers` and loads those weights (`docs/js/model.js`),
and the recursive multi-hour forecasting logic is ported line-for-line from
`src/forecast.py` to `docs/js/forecast.js`. It was verified to match the Python
model's output to float32 precision before shipping.

**Re-export after retraining** — if you retrain the model, regenerate the site's data:

```
C:\Users\mukmi\pyenvs\lstm\Scripts\python.exe -m src.export_web
```

This overwrites `docs/model/weights.json`, `docs/data/*.json` (scalers, full hourly
history, and precomputed dashboard aggregates).

**Publish it:**

1. Create a new repository on GitHub (public, so Pages can serve for free).
2. From this folder:
   ```
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source: "Deploy from a
   branch"**, then **Branch: `main`, folder: `/docs`** → Save.
4. Your site will be live at `https://<your-username>.github.io/<repo-name>/`
   within a minute or two.

**Before making the repo public**, note that `docs/data/history.json` contains your
full historical electricity + weather data (Nov 2022–Aug 2023), and it's necessarily
public once the site is live — that's what lets the predictor run without a backend.
If you'd rather not publish the raw multi-file project (the 28 MB legacy notebook,
the original `data/*.csv|xlsx`, etc.) alongside it, you can push only `docs/` to its
own repo instead of this whole folder.
