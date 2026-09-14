(function () {
  const root = document.documentElement;
  const toggle = document.getElementById('themeToggle');
  if (!toggle) return;

  function apply(theme) {
    if (theme === 'light') root.setAttribute('data-theme', 'light');
    else root.removeAttribute('data-theme');
  }

  let current = (function () {
    try { return localStorage.getItem('theme') || 'dark'; } catch (e) { return 'dark'; }
  })();
  apply(current);

  toggle.addEventListener('click', function () {
    current = current === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('theme', current); } catch (e) {}
    // Charts bake their colors in from CSS vars at creation time, so a full
    // reload is the simplest way to keep every chart in sync with the theme
    // (the inline script in base.html applies the saved theme before paint,
    // so there's no flash of the wrong theme on reload).
    location.reload();
  });
})();

// Shared chart color helpers, read live from CSS custom properties so charts
// follow the light/dark toggle without re-fetching data.
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartPalette() {
  return {
    accent: cssVar('--accent'),
    fg: cssVar('--fg'),
    fgDim: cssVar('--fg-dim'),
    fgFaint: cssVar('--fg-faint'),
    line: cssVar('--line'),
    bad: cssVar('--bad'),
    good: cssVar('--good'),
  };
}

Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
