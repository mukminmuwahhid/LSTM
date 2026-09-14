(function () {
  const startInput = document.getElementById('startInput');
  const horizonInput = document.getElementById('horizonInput');
  const horizonVal = document.getElementById('horizonVal');
  const runBtn = document.getElementById('runBtn');
  const errorBox = document.getElementById('errorBox');
  const rangeNote = document.getElementById('rangeNote');
  const chartNote = document.getElementById('chartNote');
  const resultMetrics = document.getElementById('resultMetrics');
  const hoursTableBody = document.querySelector('#hoursTable tbody');

  let bounds = null;
  let chart = null;

  function toLocalInputValue(iso) {
    // datetime-local wants "YYYY-MM-DDTHH:MM" with no timezone conversion
    return iso.slice(0, 16);
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.toggle('show', !!msg);
  }

  async function loadBounds() {
    const res = await fetch('/api/valid-range');
    bounds = await res.json();
    startInput.min = toLocalInputValue(bounds.min_start);
    startInput.max = toLocalInputValue(bounds.max_end);
    const defaultStart = new Date(bounds.max_end);
    defaultStart.setHours(defaultStart.getHours() - 47);
    startInput.value = toLocalInputValue(defaultStart.toISOString());
    rangeNote.textContent = `Valid range: ${bounds.min_start.slice(0, 16).replace('T', ' ')} → ${bounds.max_end.slice(0, 16).replace('T', ' ')}`;
  }

  horizonInput.addEventListener('input', () => {
    horizonVal.textContent = `${horizonInput.value}h`;
  });

  function renderChart(points) {
    const p = chartPalette();
    const labels = points.map((pt) => pt.timestamp.slice(0, 16).replace('T', ' '));
    const predicted = points.map((pt) => pt.predicted);
    const actual = points.map((pt) => pt.actual);
    const hasActual = actual.some((v) => v !== null);

    if (chart) chart.destroy();
    chart = new Chart(document.getElementById('forecastChart'), {
      type: 'line',
      data: {
        labels,
        datasets: [
          ...(hasActual ? [{
            label: 'Actual', data: actual, borderColor: p.fg, backgroundColor: p.fg,
            pointRadius: 0, borderWidth: 1.5, tension: 0.15, spanGaps: true,
          }] : []),
          {
            label: 'Predicted', data: predicted, borderColor: p.accent, backgroundColor: p.accent,
            pointRadius: 2, borderWidth: 2, tension: 0.15,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { color: p.fgDim, usePointStyle: true } } },
        scales: {
          x: { grid: { color: p.line }, ticks: { color: p.fgFaint, maxTicksLimit: 12, maxRotation: 0 } },
          y: { grid: { color: p.line }, ticks: { color: p.fgFaint }, title: { display: true, text: 'electricity (kWh)', color: p.fgFaint } },
        },
      },
    });
  }

  function renderTable(points) {
    hoursTableBody.innerHTML = '';
    points.forEach((pt) => {
      const err = pt.actual !== null ? Math.abs(pt.actual - pt.predicted) : null;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${pt.timestamp.slice(0, 16).replace('T', ' ')}</td>
        <td>${pt.predicted.toFixed(3)}</td>
        <td>${pt.actual !== null ? pt.actual.toFixed(3) : '—'}</td>
        <td>${err !== null ? err.toFixed(3) : '—'}</td>
      `;
      hoursTableBody.appendChild(tr);
    });
  }

  async function runForecast() {
    showError('');
    if (!startInput.value) { showError('Pick a start hour first.'); return; }

    runBtn.disabled = true;
    runBtn.innerHTML = 'Forecasting <span class="loading-dots"><span></span><span></span><span></span></span>';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: startInput.value, horizon: Number(horizonInput.value) }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed.');

      renderChart(data.points);
      renderTable(data.points);
      chartNote.textContent = `${data.points.length}h forecast starting ${startInput.value.replace('T', ' ')}`;

      const cells = resultMetrics.querySelectorAll('.stat-cell');
      if (data.metrics) {
        resultMetrics.style.display = '';
        cells[0].querySelector('.stat-value').textContent = `${data.metrics.mae.toFixed(3)} kWh`;
        cells[1].querySelector('.stat-value').textContent = `${data.metrics.rmse.toFixed(3)} kWh`;
        cells[2].querySelector('.stat-value').textContent = `${data.points.length}h`;
      } else {
        resultMetrics.style.display = '';
        cells[0].querySelector('.stat-value').textContent = '—';
        cells[1].querySelector('.stat-value').textContent = '—';
        cells[2].querySelector('.stat-value').textContent = `${data.points.length}h`;
      }
    } catch (err) {
      showError(err.message);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = 'Forecast →';
    }
  }

  runBtn.addEventListener('click', runForecast);
  loadBounds().then(runForecast).catch((err) => showError(err.message));
})();
