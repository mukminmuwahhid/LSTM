(function () {
  const fmt = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n)) ? '—' : Number(n).toFixed(d);

  function gridLineColor() { return cssVar('--line'); }
  function baseGridOpts() {
    return {
      x: { grid: { color: gridLineColor() }, ticks: { color: cssVar('--fg-faint'), maxRotation: 0 } },
      y: { grid: { color: gridLineColor() }, ticks: { color: cssVar('--fg-faint') } },
    };
  }

  async function getJSON(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
    return res.json();
  }

  function setStat(cell, value, note) {
    cell.querySelector('.stat-value').innerHTML = value;
    if (note) {
      let n = cell.querySelector('.stat-note');
      if (!n) { n = document.createElement('div'); n.className = 'stat-note'; cell.appendChild(n); }
      n.textContent = note;
    }
  }

  function renderRankList(container, labels, values, opts) {
    opts = opts || {};
    const maxAbs = Math.max(...values.map((v) => Math.abs(v)), 1e-9);
    container.innerHTML = '';
    labels.forEach((label, i) => {
      const v = values[i];
      const pct = Math.max(4, (Math.abs(v) / maxAbs) * 100);
      const row = document.createElement('div');
      row.className = 'rank-row';
      row.innerHTML = `
        <div class="rank-label" title="${label}">${prettifyLabel(label)}</div>
        <div class="rank-track"><div class="rank-fill ${v < 0 ? 'neg' : ''}" style="width:${pct}%"></div></div>
        <div class="rank-val">${opts.pct ? (v * 100).toFixed(1) + '%' : v.toFixed(opts.decimals || 3)}</div>
      `;
      container.appendChild(row);
    });
  }

  function prettifyLabel(label) {
    return label
      .replace(/electricity_/, '')
      .replace(/_/g, ' ')
      .replace(/\broll\b/, 'rolling')
      .replace(/\bstd\b/, 'std.')
      .replace(/\blag (\d+)\b/, 'lag $1h');
  }

  async function loadOverview() {
    const data = await getJSON('/api/overview');
    const cells = document.querySelectorAll('#statGrid .stat-cell');
    setStat(cells[0], data.n_rows.toLocaleString());
    setStat(cells[1], `${data.date_start.slice(0, 10)}<span class="unit"> to</span><br>${data.date_end.slice(0, 10)}`);
    setStat(cells[2], `${fmt(data.target_mean, 2)}<span class="unit">kWh avg</span>`, `±${fmt(data.target_std, 2)} std dev`);
    setStat(cells[3], `${fmt(data.target_max, 2)}<span class="unit">kWh</span>`, `min ${fmt(data.target_min, 2)} kWh`);

    const m = data.test_metrics;
    const mcells = document.querySelectorAll('#metricGrid .stat-cell');
    setStat(mcells[0], `${fmt(m.MAE, 3)}<span class="unit">kWh</span>`);
    setStat(mcells[1], `${fmt(m.RMSE, 3)}<span class="unit">kWh</span>`);
    setStat(mcells[2], `${fmt(m['MAPE_%'], 1)}<span class="unit">%</span>`);
    setStat(mcells[3], fmt(m.R2, 3));
  }

  async function loadTrend() {
    const data = await getJSON('/api/timeseries');
    const p = chartPalette();
    new Chart(document.getElementById('trendChart'), {
      type: 'line',
      data: {
        labels: data.timestamps,
        datasets: [
          {
            label: 'Daily max', data: data.max, borderColor: 'transparent',
            backgroundColor: p.accent + '22', fill: '+1', pointRadius: 0, tension: 0.25, order: 3,
          },
          {
            label: 'Daily min', data: data.min, borderColor: 'transparent',
            backgroundColor: 'transparent', pointRadius: 0, tension: 0.25, order: 3,
          },
          {
            label: 'Daily mean', data: data.mean, borderColor: p.accent, backgroundColor: p.accent,
            pointRadius: 0, borderWidth: 2, tension: 0.25, order: 1,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { color: p.fgDim, usePointStyle: true } } },
        scales: { ...baseGridOpts(), x: { ...baseGridOpts().x, ticks: { ...baseGridOpts().x.ticks, maxTicksLimit: 12 } } },
      },
    });
  }

  async function loadSeasonality() {
    const data = await getJSON('/api/seasonality');
    const p = chartPalette();
    const barOpts = (labels, values, color) => ({
      type: 'bar',
      data: { labels, datasets: [{ data: values, backgroundColor: color, borderRadius: 4, maxBarThickness: 28 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: baseGridOpts(),
      },
    });
    new Chart(document.getElementById('hourChart'), barOpts(data.hourly.labels, data.hourly.values, p.accent));
    new Chart(document.getElementById('dowChart'), barOpts(data.dow.labels, data.dow.values, p.fgDim));
    new Chart(document.getElementById('monthChart'), barOpts(data.month.labels, data.month.values, p.accent));
  }

  async function loadCorrelation() {
    const data = await getJSON('/api/correlation');
    renderRankList(document.getElementById('corrList'), data.weather_time.labels.slice(0, 10), data.weather_time.values.slice(0, 10));
    return data;
  }

  async function loadImportance() {
    const data = await getJSON('/api/importance');
    if (!data.available) {
      document.getElementById('impList').innerHTML = '<div class="panel-note">Run <code>python -m src.importance</code> to compute this.</div>';
      return null;
    }
    renderRankList(document.getElementById('impList'), data.labels.slice(0, 10), data.values.slice(0, 10), { decimals: 4 });
    document.getElementById('impNote').textContent = `baseline MAE ${fmt(data.baseline_mae, 3)} kWh`;
    return data;
  }

  async function loadTestPredictions() {
    const data = await getJSON('/api/test-predictions');
    const p = chartPalette();
    new Chart(document.getElementById('testLineChart'), {
      type: 'line',
      data: {
        labels: data.timestamps.map((t) => t.slice(5, 13)),
        datasets: [
          { label: 'Actual', data: data.actual, borderColor: p.fg, backgroundColor: p.fg, pointRadius: 0, borderWidth: 1.5, tension: 0.15 },
          { label: 'Predicted', data: data.predicted, borderColor: p.accent, backgroundColor: p.accent, pointRadius: 0, borderWidth: 2, tension: 0.15 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { color: p.fgDim, usePointStyle: true } } },
        scales: { ...baseGridOpts(), x: { ...baseGridOpts().x, ticks: { ...baseGridOpts().x.ticks, maxTicksLimit: 10 } } },
      },
    });

    new Chart(document.getElementById('testScatterChart'), {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'predicted vs actual',
          data: data.scatter_actual.map((a, i) => ({ x: a, y: data.scatter_predicted[i] })),
          backgroundColor: p.accent + '55', pointRadius: 2.5,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'actual (kWh)', color: p.fgFaint }, grid: { color: gridLineColor() }, ticks: { color: p.fgFaint } },
          y: { title: { display: true, text: 'predicted (kWh)', color: p.fgFaint }, grid: { color: gridLineColor() }, ticks: { color: p.fgFaint } },
        },
      },
    });
  }

  function setMainFactor(corrData, impData) {
    const el = document.getElementById('factorText');
    if (impData) {
      const top = impData.labels[0];
      el.innerHTML = `<strong>${prettifyLabel(top)}</strong> — permuting it degrades the model's MAE the most
        (+${fmt(impData.values[0], 4)} kWh), making it the single most relied-upon input. Time-of-day and the most
        recent reading dominate; among weather variables, solar radiation / UV index (a proxy for daytime heat and
        cooling load) matters most — direct weather has a comparatively small effect next to the strong daily
        routine of the household.`;
    } else if (corrData) {
      const top = corrData.weather_time.labels[0];
      el.innerHTML = `<strong>${prettifyLabel(top)}</strong> has the strongest raw correlation with consumption.`;
    }
  }

  Promise.all([
    loadOverview(),
    loadTrend(),
    loadSeasonality(),
    loadCorrelation(),
    loadImportance(),
    loadTestPredictions(),
  ]).then(([, , , corrData, impData]) => setMainFactor(corrData, impData))
    .catch((err) => console.error(err));
})();
