(function () {
  const startInput = document.getElementById('startInput');
  const horizonInput = document.getElementById('horizonInput');
  const horizonVal = document.getElementById('horizonVal');
  const runBtn = document.getElementById('runBtn');
  const errorBox = document.getElementById('errorBox');
  const loadError = document.getElementById('loadError');
  const loadNote = document.getElementById('loadNote');
  const rangeNote = document.getElementById('rangeNote');
  const chartNote = document.getElementById('chartNote');
  const resultMetrics = document.getElementById('resultMetrics');
  const hoursTableBody = document.querySelector('#hoursTable tbody');

  let chart = null;

  function pad(n) { return String(n).padStart(2, '0'); }
  function toLocalInputValue(d) {
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.classList.toggle('show', !!msg);
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
      const points = await ForecastEngine.recursiveForecast(startInput.value, Number(horizonInput.value));

      renderChart(points);
      renderTable(points);
      chartNote.textContent = `${points.length}h forecast starting ${startInput.value.replace('T', ' ')}`;

      const actuals = points.map((p) => p.actual);
      const cells = resultMetrics.querySelectorAll('.stat-cell');
      resultMetrics.style.display = '';
      if (actuals.every((a) => a !== null)) {
        const errs = points.map((p) => Math.abs(p.actual - p.predicted));
        const sqErrs = points.map((p) => (p.actual - p.predicted) ** 2);
        const mae = errs.reduce((a, b) => a + b, 0) / errs.length;
        const rmse = Math.sqrt(sqErrs.reduce((a, b) => a + b, 0) / sqErrs.length);
        cells[0].querySelector('.stat-value').textContent = `${mae.toFixed(3)} kWh`;
        cells[1].querySelector('.stat-value').textContent = `${rmse.toFixed(3)} kWh`;
      } else {
        cells[0].querySelector('.stat-value').textContent = '—';
        cells[1].querySelector('.stat-value').textContent = '—';
      }
      cells[2].querySelector('.stat-value').textContent = `${points.length}h`;
    } catch (err) {
      showError(err.message);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = 'Forecast →';
    }
  }

  runBtn.addEventListener('click', runForecast);

  (async function boot() {
    try {
      await ForecastEngine.init();
      const { minStart, maxEnd, maxHorizon } = ForecastEngine.getBounds();

      startInput.min = toLocalInputValue(minStart);
      startInput.max = toLocalInputValue(maxEnd);
      horizonInput.max = maxHorizon;

      const defaultStart = new Date(maxEnd.getTime() - 47 * 3600 * 1000);
      startInput.value = toLocalInputValue(defaultStart);
      rangeNote.textContent = `Valid range: ${toLocalInputValue(minStart).replace('T', ' ')} → ${toLocalInputValue(maxEnd).replace('T', ' ')}`;

      startInput.disabled = false;
      horizonInput.disabled = false;
      runBtn.disabled = false;
      loadNote.style.display = 'none';

      await runForecast();
    } catch (err) {
      loadNote.style.display = 'none';
      loadError.textContent = `Failed to load model/data: ${err.message}`;
      loadError.classList.add('show');
    }
  })();
})();
