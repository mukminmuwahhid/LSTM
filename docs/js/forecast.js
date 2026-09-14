// Client-side port of src/forecast.py — recursive multi-hour forecasting.
// Runs entirely in the browser against the static history.json + the
// TensorFlow.js model, so it works with no backend (GitHub Pages).
//
// Same approach as the Python version: to forecast beyond 1 hour, each
// prediction is fed back in as if it were an observed value, so later
// steps' lag/rolling features are built from a hybrid actual+predicted
// history — never from true future electricity. Weather/calendar features
// for the forecast window come from the historical record (no live
// weather feed), so valid forecasts are limited to the historical period.

const ForecastEngine = (function () {
  let state = null; // { history, meta, scalers, model }

  function hourMs(n) { return n * 3600 * 1000; }

  function toHourTs(d) {
    const t = new Date(d);
    t.setMinutes(0, 0, 0);
    return t;
  }

  function isoHour(d) {
    // "YYYY-MM-DDTHH:00:00" in local wall-clock time (matches the dataset,
    // which has no timezone info — it's treated as naive local hours throughout).
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:00:00`;
  }

  function timeFeatures(d) {
    const hour = d.getHours();
    const dow = (d.getDay() + 6) % 7; // JS: 0=Sun..6=Sat -> convert to Python's 0=Mon..6=Sun
    const month = d.getMonth() + 1;
    return {
      hour_sin: Math.sin((2 * Math.PI * hour) / 24),
      hour_cos: Math.cos((2 * Math.PI * hour) / 24),
      dow_sin: Math.sin((2 * Math.PI * dow) / 7),
      dow_cos: Math.cos((2 * Math.PI * dow) / 7),
      month_sin: Math.sin((2 * Math.PI * month) / 12),
      month_cos: Math.cos((2 * Math.PI * month) / 12),
      is_weekend: dow >= 5 ? 1 : 0,
    };
  }

  async function init() {
    if (state) return state;
    const [history, meta, scalers, model] = await Promise.all([
      fetch(assetUrl('data/history.json')).then((r) => r.json()),
      fetch(assetUrl('data/meta.json')).then((r) => r.json()),
      fetch(assetUrl('data/scalers.json')).then((r) => r.json()),
      loadTfjsModel(assetUrl('model/weights.json')),
    ]);

    const elecByTs = new Map();
    const weatherByTs = new Map();
    for (let i = 0; i < history.timestamps.length; i++) {
      const ts = history.timestamps[i];
      elecByTs.set(ts, history[meta.target_col][i]);
      const w = {};
      for (const col of meta.weather_cols) w[col] = history[col][i];
      weatherByTs.set(ts, w);
    }

    const minTs = new Date(history.timestamps[0]);
    const maxTs = new Date(history.timestamps[history.timestamps.length - 1]);
    const minStart = new Date(minTs.getTime() + hourMs(meta.lookback + Math.max(...meta.lag_hours)));

    state = { meta, scalers, model, elecByTs, weatherByTs, minTs, maxTs, minStart };
    return state;
  }

  function getBounds() {
    if (!state) throw new Error('ForecastEngine not initialized');
    return { minStart: state.minStart, maxEnd: state.maxTs, maxHorizon: state.meta.max_forecast_horizon };
  }

  function scaleRow(rawRow, scale, min) {
    return rawRow.map((v, j) => v * scale[j] + min[j]);
  }

  async function recursiveForecast(startInput, horizon) {
    const { meta, scalers, model, elecByTs, weatherByTs, minTs, maxTs, minStart } = state;
    const startTs = toHourTs(startInput);

    if (startTs < minStart || startTs > maxTs) {
      throw new Error(`start must be between ${isoHour(minStart)} and ${isoHour(maxTs)}`);
    }
    const endTs = new Date(startTs.getTime() + hourMs(horizon - 1));
    if (endTs > maxTs) {
      const maxHorizonHere = Math.round((maxTs - startTs) / hourMs(1)) + 1;
      throw new Error(
        `horizon too large for this start time — historical data (used as the weather-forecast stand-in) ` +
        `only extends to ${isoHour(maxTs)}. Max horizon from this start is ${maxHorizonHere}h.`
      );
    }

    const predicted = new Map(); // iso -> value
    const featCache = new Map(); // iso -> raw feature array (ordered per meta.feature_cols)

    function elecValue(d) {
      const iso = isoHour(d);
      if (predicted.has(iso)) return predicted.get(iso);
      if (elecByTs.has(iso)) return elecByTs.get(iso);
      throw new Error(`missing electricity history at ${iso}`);
    }

    function computeFeatureRow(d) {
      const iso = isoHour(d);
      if (featCache.has(iso)) return featCache.get(iso);

      const w = weatherByTs.get(iso);
      if (!w) throw new Error(`missing weather history at ${iso}`);
      const tfeat = timeFeatures(d);
      const rowDict = { ...w, ...tfeat };

      for (const lag of meta.lag_hours) {
        const refD = new Date(d.getTime() - hourMs(lag));
        rowDict[`${meta.target_col}_lag_${lag}`] = elecValue(refD);
      }
      for (const win of meta.rolling_windows) {
        const vals = [];
        for (let k = 1; k <= win; k++) vals.push(elecValue(new Date(d.getTime() - hourMs(k))));
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
        const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (vals.length - 1); // ddof=1, matches pandas
        rowDict[`${meta.target_col}_roll_mean_${win}`] = mean;
        rowDict[`${meta.target_col}_roll_std_${win}`] = Math.sqrt(variance);
      }

      const row = meta.feature_cols.map((c) => rowDict[c]);
      featCache.set(iso, row);
      return row;
    }

    const results = [];
    for (let h = 0; h < horizon; h++) {
      const targetD = new Date(startTs.getTime() + hourMs(h));
      const windowRaw = [];
      for (let k = meta.lookback; k >= 1; k--) {
        windowRaw.push(computeFeatureRow(new Date(targetD.getTime() - hourMs(k))));
      }
      const windowScaled = windowRaw.map((row) => scaleRow(row, scalers.feature_scale, scalers.feature_min));

      const X = tf.tensor([windowScaled]); // [1, lookback, n_features]
      const yScaledTensor = model.predict(X);
      const yScaled = (await yScaledTensor.data())[0];
      X.dispose();
      yScaledTensor.dispose();

      const yPred = (yScaled - scalers.target_min[0]) / scalers.target_scale[0];
      const iso = isoHour(targetD);
      predicted.set(iso, yPred);

      const actual = elecByTs.has(iso) ? elecByTs.get(iso) : null;
      results.push({ timestamp: iso, predicted: Math.round(yPred * 10000) / 10000, actual });
    }

    return results;
  }

  return { init, getBounds, recursiveForecast, isoHour, toHourTs };
})();
