/* app.js — Sales Analysis Agent UI logic */

const titles = {
  overview:  ["Data Overview",  "Data quality and revenue summary"],
  schema:    ["Schema AI",      "Automatic column name mapping"],
  kpis:      ["KPIs",           "Key performance indicators"],
  anomalies: ["Anomalies",      "Unusual patterns detected in the data"],
  forecast:  ["Forecast",       "Projected revenue for the next 3 months"],
  chat:      ["Chat",           "Ask plain English questions about your data"],
};

let loaded = false;
let kpiData = null;
let forecastHistory = [];

//function for uploading csv and auto-running pipeline
async function uploadCSV() {         
  const fileInput = document.getElementById('csv-upload');
  if (!fileInput.files.length) return;

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);

  const uploadBtn = document.getElementById('upload-btn');
  const fileNameDisplay = document.getElementById('upload-filename');

  fileNameDisplay.textContent = file.name;
  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Uploading...';

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    loaded = false;
    document.getElementById('load-btn').disabled = false;
    document.getElementById('load-btn').textContent = '▶ Load ERP Data';

    // Auto-run pipeline after successful upload
    await loadData();

  } catch (err) {
    alert('Upload failed: ' + err.message);
    fileNameDisplay.textContent = 'Upload failed';
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = 'Upload CSV';
    fileInput.value = ''; // reset selection
  }
}

// ── Navigation ────────────────────────────────────────────────────────────────

function showPage(page) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + page).classList.add('active');
  document.querySelector(`[data-page="${page}"]`).classList.add('active');
  document.getElementById('page-title').textContent = titles[page][0];
  document.getElementById('page-sub').textContent   = titles[page][1];
}

// ── Load pipeline ─────────────────────────────────────────────────────────────

async function loadData() {
  if (loaded) return;
  setStatus('loading', 'Running pipeline…');
  document.getElementById('loading').classList.add('show');
  document.getElementById('load-btn').disabled = true;

  try {
    const res  = await fetch('/api/load', { method: 'POST' });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    renderSchemaTable(data.mappings);
    renderOverviewStats(data);

    const kres = await fetch('/api/kpis');
    kpiData = await (kres).json();
    renderKPIs(kpiData, data);

    const ares = await fetch('/api/anomalies');
    const adat = await ares.json();
    renderAnomalies(adat.anomalies);

    const fres = await fetch('/api/forecast');
    const fdat = await fres.json();
    
    // Sync the select dropdown back to 3 months default upon fresh load
    document.getElementById('forecast-horizon').value = "3";
    document.getElementById('forecast-title-text').textContent = '3-Month Sales Forecast';
    renderForecast(fdat.forecast, kpiData);

    // Forecast Intelligence Engine (Phase 1) — dataset analysis panel.
    // Failure here must never block the rest of the pipeline from loading.
    await loadForecastAnalysis();

    // Phase 5 — populate the forecast-level selector from this
    // dataset's own dimensions. Lightweight (value counts only) and
    // non-fatal: on failure the selector stays at Company Total.
    await loadGroupMetadata();

    loaded = true;
    setStatus('ready', 'Data loaded');
    document.getElementById('overview-empty').style.display   = 'none';
    document.getElementById('overview-content').style.display = 'block';
    document.getElementById('load-btn').textContent = '✓ Loaded';

    // Load duplicates automatically
    await loadDuplicates();

  } catch (e) {
    setStatus('idle', 'Error — check console');
    alert('Error: ' + e.message);
    document.getElementById('load-btn').disabled = false;
  } finally {
    document.getElementById('loading').classList.remove('show');
  }
}

function setStatus(cls, text) {
  const dot  = document.getElementById('status-dot');
  const span = document.getElementById('status-text');
  dot.className  = 'status-dot ' + cls;
  span.textContent = text;
}

// ── Overview ──────────────────────────────────────────────────────────────────

function renderOverviewStats(data) {
  const g = document.getElementById('overview-summary-grid');
  g.innerHTML = '';
  const stats = [
    { label: 'Total Rows',          value: fmtNum(data.rows),       sub: 'in ERP export',        icon: '▤', clickable: false },
    { label: 'Columns Mapped',      value: data.columns,            sub: 'auto-identified',      icon: '⊞', clickable: false },
    { label: 'Rows After Cleaning', value: fmtNum(data.rows_clean), sub: 'clean rows',           icon: '✓', clickable: false },
    { label: 'Duplicates Removed',  value: data.dupes,              sub: 'by cleaning pipeline', icon: '⊘',
      clickable: data.dupes > 0 },
  ];
  stats.forEach(s => {
    const d = document.createElement('div');
    d.className = 'stat-card' + (s.clickable ? ' stat-card-clickable' : '');
    if (s.clickable) d.onclick = () => scrollToDupes();
    d.innerHTML = `
      <div class="stat-icon">${s.icon}</div>
      <div class="stat-label">${s.label}</div>
      <div class="stat-value">${s.value}</div>
      <div class="stat-sub">${s.sub}${s.clickable ? ' <span class="stat-link">View →</span>' : ''}</div>
    `;
    g.appendChild(d);
  });
}

function scrollToDupes() {
  const panel = document.getElementById('dupes-panel');
  const card  = document.getElementById('dupes-card');
  if (panel.style.display === 'none' && _allDupes.length > 0) {
    panel.style.display = 'block';
    document.getElementById('dupes-toggle-btn').textContent = 'Hide duplicates';
  }
  setTimeout(() => card.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
}

// ── Duplicate records ─────────────────────────────────────────────────────────

let _allDupes   = [];
let _dupesCols  = [];
let _dupesIdCol = null;
let _selectedInvNo = null;
let _selectedCount = 0;
let _allGroups  = [];

async function loadDuplicates() {
  const res  = await fetch('/api/duplicates');
  const data = await res.json();

  if (!data.ok) return;

  // Backend returns flat duplicate rows
  _allDupes   = data.duplicates || [];
  _dupesCols  = data.columns || [];
  _dupesIdCol = data.id_col;

  // Build grouped structure on frontend
  _allGroups = buildGroups(_allDupes);

  document.getElementById('dupes-badge').textContent = data.total;

  if (data.total === 0) {
    document.getElementById('dupes-summary').textContent =
      'No duplicate records found — data is clean.';
    document.getElementById('dupes-toggle-btn').style.display = 'none';
    return;
  }

  document.getElementById('dupes-summary').textContent =
    `${data.total} rows share a duplicate invoice number. Each group below shows the matching rows side by side.`;

  renderDupeGroups(_allGroups);
}

function buildGroups(rows) {
  const map = {};
  rows.forEach(r => {
    const inv = r[_dupesIdCol];
    if (!map[inv]) map[inv] = [];
    map[inv].push(r);
  });
  return Object.entries(map)
    .filter(([, rows]) => rows.length > 1)
    .map(([invNo, rows]) => ({ invNo, rows }));
}

function renderDupeGroups(groups) {
  const container = document.getElementById('dupes-groups');
  container.innerHTML = '';

  if (groups.length === 0) {
    container.innerHTML = '<p style="color:var(--text-3);font-size:13px;padding:12px 0">No duplicates match your filter.</p>';
    return;
  }

  const showCols = _dupesCols.slice(0, 7);

  groups.forEach(group => {

const invNo = group.inv_no;
const rows = group.rows;
    const block = document.createElement('div');
    block.className = 'dupe-group-block';

    const header = document.createElement('div');
    header.className = 'dupe-group-header';
    header.innerHTML = `
<div class="dupe-group-inv">

  <span class="dupe-inv-label">
    Invoice
  </span>

  <code class="dupe-inv-code">
    ${invNo}
  </code>

  <span class="dupe-count-pill">
    ${rows.length} copies
  </span>

<span class="dupe-confidence-pill exact">
    Exact Duplicate
</span>

</div>

<button
 class="btn-sm-outline dupe-manage-btn"
 onclick="openDupeModal('${invNo}', ${rows.length})">

 Manage

</button>
`;
    block.appendChild(header);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'dupe-side-table-wrap';

    const table = document.createElement('table');
    table.className = 'dupe-side-table';

    const thead = document.createElement('thead');
    thead.innerHTML = '<tr><th class="dupe-field-col">Field</th>' +
      rows.map((_, i) => `<th class="dupe-copy-col ${i===0?'dupe-first-copy':'dupe-extra-copy'}">Copy ${i+1}</th>`).join('') +
      '</tr>';
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    showCols.forEach(col => {
      const tr = document.createElement('tr');
      const vals = rows.map(r => String(r[col] !== undefined ? r[col] : '—'));
      const allSame = vals.every(v => v === vals[0]);
      tr.innerHTML = `<td class="dupe-field-name">${col}</td>` +
        rows.map((r, i) => {
          const val = r[col] !== undefined ? r[col] : '—';
          return `<td class="${i===0 ? 'dupe-first-val' : 'dupe-extra-val'} ${!allSame ? 'dupe-val-diff' : ''}">${val}</td>`;
        }).join('');
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    block.appendChild(tableWrap);
    container.appendChild(block);
  });
}

function filterDupes(query) {
  if (!query.trim()) { renderDupeGroups(_allGroups); return; }
  const q = query.toLowerCase();
  const filtered = _allGroups.filter(g =>
    g.invNo.toLowerCase().includes(q) ||
    g.rows.some(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)))
  );
  renderDupeGroups(filtered);
}

function toggleDuplicates() {
  const panel = document.getElementById('dupes-panel');
  const btn   = document.getElementById('dupes-toggle-btn');
  if (panel.style.display === 'none') {
    if (_allDupes.length === 0) { loadDuplicates(); return; }
    renderDupeGroups(_allGroups);
    panel.style.display = 'block';
    btn.textContent = 'Hide duplicates';
  } else {
    panel.style.display = 'none';
    btn.textContent = 'View duplicates';
  }
}

function openDupeModal(invNo, count) {
  _selectedInvNo = invNo;
  _selectedCount = count;
  document.getElementById('modal-inv-no').textContent = invNo;
  document.getElementById('modal-count').textContent  = count;
  document.getElementById('modal-msg').textContent    = '';
  document.getElementById('dupe-modal-overlay').style.display = 'flex';
}

function closeDupeModal() {
  document.getElementById('dupe-modal-overlay').style.display = 'none';
}

async function handleDupe(action) {
  const msg = document.getElementById('modal-msg');
  msg.textContent = 'Processing…';
  msg.style.color = 'var(--text-2)';

  const res  = await fetch('/api/duplicates/delete', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({
      inv_no:     _selectedInvNo,
      delete_all: action === 'delete_all',
    }),
  });
  const data = await res.json();

  if (data.ok) {
    msg.textContent = `✓ Removed ${data.removed} row(s) for ${data.inv_no}.`;
    msg.style.color = 'var(--green)';
    setTimeout(async () => {
      closeDupeModal();
      await loadDuplicates();
    }, 800);
  } else {
    msg.textContent = 'Error: ' + data.error;
    msg.style.color = 'var(--red)';
  }
}

// ── Schema table ──────────────────────────────────────────────────────────────

function renderSchemaTable(mappings) {
  const tbody  = document.getElementById('schema-tbody');
  const badge  = document.getElementById('schema-badge');
  const table  = document.getElementById('schema-table');
  const empty  = document.getElementById('schema-empty');
  const panel  = document.getElementById('correct-panel');
  const sel    = document.getElementById('correct-col');

  badge.textContent = mappings.length + ' columns';
  tbody.innerHTML   = '';
  sel.innerHTML     = '<option value="">Select column…</option>';

  mappings.forEach(m => {
    const conf   = m.confidence;
    const cls    = conf >= 0.85 ? 'high' : conf >= 0.65 ? 'medium' : 'low';
    const pct    = Math.round(conf * 100) + '%';
    
    // Process the new reasoning array
    let reasonsHtml = `<div style="color:var(--text-3);font-size:11px;margin-bottom:4px;font-weight:600;text-transform:uppercase;">${m.source}</div>`;
    if (m.reasons && m.reasons.length > 0) {
      reasonsHtml += `<ul style="margin:0; padding-left:14px; font-size:11px; color:var(--text-2); line-height:1.4;">
        ${m.reasons.map(r => `<li>${r}</li>`).join('')}
      </ul>`;
    }

    const tr     = document.createElement('tr');
    tr.innerHTML = `
      <td><code style="font-size:12px;background:#f3f4f6;padding:2px 6px;border-radius:4px">${m.col}</code></td>
      <td style="font-weight:500">${m.name}</td>
      <td><span class="cat-pill">${m.category}</span></td>
      <td><span class="conf-pill conf-${cls}">${pct}</span></td>
      <td>${reasonsHtml}</td>
      <td><button class="btn-edit" onclick="prefillCorrect('${m.col}')">Edit</button></td>
    `;
    tbody.appendChild(tr);
    
    const opt = document.createElement('option');
    opt.value = m.col; opt.textContent = m.col + '  →  ' + m.name;
    sel.appendChild(opt);
  });

  table.style.display = 'table';
  empty.style.display = 'none';
  panel.style.display = 'block';
}

function prefillCorrect(col) {
  showPage('schema');
  document.getElementById('correct-col').value = col;
  document.getElementById('correct-name').focus();
}

async function submitCorrection() {
  const col  = document.getElementById('correct-col').value;
  const name = document.getElementById('correct-name').value.trim();
  const msg  = document.getElementById('correct-msg');
  if (!col || !name) { msg.textContent = 'Please select a column and enter a name.'; msg.style.color = 'var(--red)'; return; }

  const res  = await fetch('/api/correct', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ column: col, name }),
  });
  const data = await res.json();
  if (data.ok) {
    msg.textContent = '✓ ' + data.message + ' — reload to see updated table.';
    msg.style.color = 'var(--green)';
    document.getElementById('correct-name').value = '';
  } else {
    msg.textContent = 'Error: ' + data.error;
    msg.style.color = 'var(--red)';
  }
}

// ── KPIs ──────────────────────────────────────────────────────────────────────

function renderKPIs(k, data) {
  document.getElementById('kpi-empty').style.display   = 'none';
  document.getElementById('kpi-content').style.display = 'block';

  const g = document.getElementById('kpi-stat-grid');
  g.innerHTML = '';
  [
    { label: 'Total Revenue',       value: fmtINR(k.total_revenue),    sub: 'net sales value' },
    { label: 'Avg Order Value',     value: fmtINR(k.avg_order_value),  sub: 'per invoice' },
    { label: 'Unique Customers',    value: k.unique_customers,         sub: 'active buyers' },
    { label: 'Total Units Sold',    value: fmtNum(k.total_units),      sub: 'units' },
    { label: 'Total Discount',      value: fmtINR(k.total_discount),   sub: k.disc_pct + '% of revenue' },
    { label: 'Overdue Amount',      value: fmtINR(k.overdue_amount),   sub: 'unpaid invoices' },
  ].forEach(s => g.appendChild(statCard(s.label, s.value, s.sub)));

  const tc = document.getElementById('top-customers');
  tc.innerHTML = '';
  Object.entries(k.top_customers).forEach(([id, val], i) => {
    tc.appendChild(listRow(`${i+1}. ${id}`, fmtINR(val)));
  });

  const ps = document.getElementById('payment-status');
  ps.innerHTML = '';
  const colors = { PAID: 'var(--green)', PENDING: 'var(--amber)', OVERDUE: 'var(--red)' };
  Object.entries(k.payment_status).forEach(([status, count]) => {
    const row = listRow(status, count + ' invoices');
    row.querySelector('.list-label').style.color = colors[status] || 'var(--text-2)';
    ps.appendChild(row);
  });

  renderBars('sp-bars', k.revenue_by_category, true);

  const cs = document.getElementById('clean-summary');
  cs.innerHTML = '';
  [
    ['Rows loaded',     fmtNum(data.rows)],
    ['Rows after clean',fmtNum(data.rows_clean)],
    ['Duplicates removed', data.dupes],
  ].forEach(([l, v]) => cs.appendChild(listRow(l, v)));
}

// ── Anomalies ─────────────────────────────────────────────────────────────────

function renderAnomalies(anomalies) {
  document.getElementById('anomaly-empty').style.display = 'none';
  const list = document.getElementById('anomaly-list');
  list.style.display = 'block';
  list.innerHTML = '';

  if (!anomalies || anomalies.length === 0) {
    list.innerHTML = '<div class="card" style="text-align:center;padding:40px;color:var(--text-3)">No anomalies detected — data looks clean.</div>';
    return;
  }

  anomalies.forEach(a => {
    const card = document.createElement('div');
    card.className = 'anomaly-card ' + a.severity;
    card.innerHTML = `
      <div class="anomaly-header">
        <span class="sev-badge sev-${a.severity}">${a.severity}</span>
        <div class="anomaly-title">${a.type}</div>
      </div>
      <div class="anomaly-desc">${a.description}</div>
    `;
    list.appendChild(card);
  });
}

// ── Forecast (Section 1 hero + Section 2 chart + Section 3 monthly cards) ──
// Redesign note: this renders the baseline forecast (existing
// analytics.run_forecast(), fetched via GET/POST /api/forecast exactly as
// before — unchanged backend call). Once the user runs the Auto
// Forecasting Engine, renderAutoForecast() below repaints these same
// elements with the richer, model-selected result — same DOM targets, so
// nothing needs to be duplicated on the page.

let autoForecastActive = false;

function renderForecast(forecast, k) {
  document.getElementById('forecast-empty').style.display  = 'none';
  document.getElementById('forecast-content').style.display = 'block';

  const horizonLabel = forecast.length + (forecast.length === 1 ? ' Month' : ' Months');
  document.getElementById('fc-hero-horizon').textContent = horizonLabel;

  const total = forecast.reduce((sum, f) => sum + f.value, 0);
  document.getElementById('fc-hero-revenue').textContent = fmtINR(total);
  document.getElementById('fc-hero-revenue-sub').textContent = `total across ${forecast.length} month${forecast.length === 1 ? '' : 's'}`;

  renderMonthCards(forecast);

  const histEntries = Object.entries(k.monthly_revenue).slice(-6);
  forecastHistory = histEntries;
  renderCombinedChart('forecast-chart', histEntries, forecast.map(f => ({ label: f.month, value: f.value })), null);
}

async function updateForecast() {
  const horizon = document.getElementById('forecast-horizon').value;
  const label = horizon + (horizon === '1' ? ' Month' : ' Months');
  document.getElementById('fc-hero-horizon').textContent = label;

  // Once the auto engine has been run, changing the horizon re-runs it
  // (rather than silently falling back to the simpler baseline forecast)
  // so the dashboard stays in a consistent, model-selected state.
  if (autoForecastActive) {
    await runAutoForecast();
    return;
  }

  try {
    const res = await fetch('/api/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ months: parseInt(horizon) })
    });
    const fdat = await res.json();
    if (!fdat.ok) throw new Error(fdat.error);
    renderForecast(fdat.forecast, kpiData);
  } catch (err) {
    alert('Failed to update forecast: ' + err.message);
  }
}

// Section 3 — compact monthly result cards, with an expandable detail table.
function renderMonthCards(forecast) {
  const grid = document.getElementById('fc-month-grid');
  grid.innerHTML = '';
  forecast.forEach(f => {
    const card = document.createElement('div');
    card.className = 'fc-month-card';
    card.innerHTML = `
      <div class="fc-month-label">${monthLabel(f.period || f.month)}</div>
      <div class="fc-month-value">${fmtINR(f.value)}</div>
    `;
    grid.appendChild(card);
  });

  const tbody = document.getElementById('fc-detail-tbody');
  tbody.innerHTML = '';
  forecast.forEach(f => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${f.period || f.month}</td><td>${fmtINR(f.value)}</td>`;
    tbody.appendChild(tr);
  });
}

function monthLabel(periodStr) {
  const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  const parts = String(periodStr).split('-');
  if (parts.length !== 2) return periodStr;
  const idx = parseInt(parts[1], 10) - 1;
  if (idx < 0 || idx > 11) return periodStr;
  return `${MONTHS[idx]} ${parts[0]}`;
}

function toggleForecastTable() {
  const wrap = document.getElementById('fc-table-wrap');
  const btn  = document.getElementById('fc-table-toggle');
  const showing = wrap.style.display !== 'none';
  wrap.style.display = showing ? 'none' : 'block';
  btn.textContent = showing ? 'View Detailed Forecast Table' : 'Hide Detailed Forecast Table';
}

// Section 5/6 — collapsed-by-default accordions for technical detail.
function toggleAccordion(id) {
  document.getElementById(id).classList.toggle('open');
}

// ── Chat ──────────────────────────────────────────────────────────────────────

function prefill(text) {
  document.getElementById('chat-input').value = text;
  showPage('chat');
  sendQuestion();
}

async function sendQuestion() {
  const input = document.getElementById('chat-input');
  const q = input.value.trim();
  if (!q) return;
  if (!loaded) { appendMsg('agent', 'Please load the ERP data first by clicking "Load ERP Data".'); return; }

  appendMsg('user', q);
  input.value = '';

  const res  = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: q }),
  });
  const data = await res.json();
  appendMsg('agent', data.ok ? data.answer : 'Sorry, I could not answer that. Try rephrasing.');
}

function appendMsg(role, text) {
  const box  = document.getElementById('chat-messages');
  const div  = document.createElement('div');
  div.className = 'chat-msg ' + role;
  div.innerHTML = `<div class="msg-bubble">${text}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

// ── Chart helpers ─────────────────────────────────────────────────────────────

function renderBars(containerId, data, money) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  if (!data || !Object.keys(data).length) { el.textContent = 'No data'; return; }
  const max = Math.max(...Object.values(data));
  Object.entries(data).forEach(([label, val]) => {
    const pct = max > 0 ? (val / max * 100).toFixed(1) : 0;
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="bar-label" title="${label}">${label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <span class="bar-val">${money ? fmtINR(val) : val}</span>
    `;
    el.appendChild(row);
  });
}

function renderCombinedChart(containerId, hist, forecast, bandPct) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  const hLabels = hist.map(e => e[0]);
  const hVals   = hist.map(e => e[1]);
  drawLineChart(el, hLabels, hVals, forecast, '#2563EB', bandPct);
}

// Redesign: taller chart (Section 2 is now the dominant element on the
// page), an optional shaded confidence band around the forecast line
// (bandPct = fractional +/- range, e.g. 0.15; pass null/undefined to
// render no band), and hover tooltips on every point via a shared
// tooltip element (#fc-chart-tooltip, positioned by showChartTooltip()).
function drawLineChart(container, labels, values, fcPoints, color, bandPct) {
  const W = Math.max(container.clientWidth || 600, (labels.length + fcPoints.length) * 60 + 60);
  const H = 340;
  const PL = 64, PR = 20, PT = 20, PB = 40;
  const chartW = W - PL - PR;
  const chartH = H - PT - PB;

  // Phase 4: when a forecast point carries real prediction bounds
  // (f.lower / f.upper, populated from the API's lower_80/upper_80), the
  // band is drawn from THOSE. `bandPct` remains supported only for the
  // baseline /api/forecast view, which has no validated model behind it
  // and therefore no interval to draw.
  const hasBounds = fcPoints.some(f => f.lower != null && f.upper != null);
  const bandVals = hasBounds
    ? fcPoints.flatMap(f => [f.upper != null ? f.upper : f.value, f.lower != null ? f.lower : f.value])
    : (bandPct ? fcPoints.flatMap(f => [f.value * (1 + bandPct), f.value * (1 - bandPct)]) : []);
  const allVals = [...values, ...fcPoints.map(f => f.value), ...bandVals];
  const maxV = Math.max(...allVals, 1) * 1.1;
  const minV = 0;
  const range = maxV - minV || 1;

  const xs = labels.map((_, i) => PL + (i / (labels.length - 1 || 1)) * chartW);
  const ys = values.map(v => PT + chartH - ((v - minV) / range) * chartH);

  const fcXStart = labels.length;
  const totalPts = labels.length + fcPoints.length;
  const fcXs = fcPoints.map((_, i) => PL + ((fcXStart + i) / (totalPts - 1 || 1)) * chartW);
  const fcYs = fcPoints.map(f => PT + chartH - ((f.value - minV) / range) * chartH);
  const toY = v => PT + chartH - ((v - minV) / range) * chartH;

  const histPts = xs.map((x,i) => `${x},${ys[i]}`).join(' ');
  const fcPts   = (xs.length > 0 && fcXs.length > 0)
    ? `${xs[xs.length-1]},${ys[ys.length-1]} ` + fcXs.map((x,i) => `${x},${fcYs[i]}`).join(' ')
    : '';

  const areaHist = xs.length ? `${PL},${PT+chartH} ` + histPts + ` ${xs[xs.length-1]},${PT+chartH}` : '';

  // Confidence band: a shaded ribbon around the forecast line, anchored to
  // the last historical point so it joins the chart with no visual gap.
  let bandPolygon = '';
  if ((hasBounds || bandPct) && fcXs.length > 0) {
    const anchorX = xs.length ? xs[xs.length - 1] : fcXs[0];
    const anchorY = ys.length ? ys[ys.length - 1] : fcYs[0];
    const upperOf = i => hasBounds
      ? (fcPoints[i].upper != null ? fcPoints[i].upper : fcPoints[i].value)
      : fcPoints[i].value * (1 + bandPct);
    const lowerOf = i => hasBounds
      ? (fcPoints[i].lower != null ? fcPoints[i].lower : fcPoints[i].value)
      : fcPoints[i].value * (1 - bandPct);
    const upperPts = [`${anchorX},${anchorY}`].concat(fcXs.map((x, i) => `${x},${toY(upperOf(i))}`));
    const lowerPts = [`${anchorX},${anchorY}`].concat(fcXs.map((x, i) => `${x},${toY(lowerOf(i))}`));
    bandPolygon = `<polygon points="${upperPts.join(' ')} ${lowerPts.slice().reverse().join(' ')}" fill="#7C3AED" opacity="0.10"/>`;
  }

  const gridLines = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const y = PT + chartH - f * chartH;
    const v = (minV + f * range);
    return `<line x1="${PL}" y1="${y}" x2="${W-PR}" y2="${y}" stroke="#EEF1F5" stroke-width="1"/>
            <text x="${PL-8}" y="${y+4}" font-size="11" fill="#9CA3AF" text-anchor="end">${fmtShort(v)}</text>`;
  }).join('');

  const xLabels = labels.map((l,i) => {
    const x = xs[i];
    return `<text x="${x}" y="${H-10}" font-size="11" fill="#9CA3AF" text-anchor="middle">${String(l).slice(5)}</text>`;
  }).join('') + fcPoints.map((f,i) => {
    const x = fcXs[i];
    return `<text x="${x}" y="${H-10}" font-size="11" fill="#7C3AED" text-anchor="middle">${String(f.label).slice(5)}</text>`;
  }).join('');

  const dots = xs.map((x,i) => `<circle cx="${x}" cy="${ys[i]}" r="4" fill="${color}" stroke="#fff" stroke-width="1.5"
    onmouseenter="showChartTooltip(event,'${labels[i]}','${fmtINR(values[i])}','Historical')" onmouseleave="hideChartTooltip()"/>`).join('');
  const fcDots = fcXs.map((x,i) => `<circle cx="${x}" cy="${fcYs[i]}" r="4" fill="#7C3AED" stroke="#fff" stroke-width="1.5"
    onmouseenter="showChartTooltip(event,'${fcPoints[i].label}','${fmtINR(fcPoints[i].value)}','Forecast')" onmouseleave="hideChartTooltip()"/>`).join('');

  container.innerHTML = `
    <svg class="chart-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
      <defs>
        <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="${color}" stop-opacity=".15"/>
          <stop offset="100%" stop-color="${color}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${gridLines}
      ${histPts ? `<polygon points="${areaHist}" fill="url(#grad)"/>` : ''}
      ${bandPolygon}
      ${histPts ? `<polyline points="${histPts}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round"/>` : ''}
      ${fcPts   ? `<polyline points="${fcPts}"   fill="none" stroke="#7C3AED" stroke-width="2.5" stroke-dasharray="6 4" stroke-linejoin="round"/>` : ''}
      ${dots}${fcDots}
      ${xLabels}
    </svg>
  `;
}

function showChartTooltip(evt, label, value, kind) {
  const tip = document.getElementById('fc-chart-tooltip');
  if (!tip) return;
  const outer = tip.parentElement.getBoundingClientRect();
  tip.innerHTML = `<div class="fc-tt-kind">${kind}</div><div class="fc-tt-label">${label}</div><div class="fc-tt-value">${value}</div>`;
  tip.style.left = (evt.clientX - outer.left + 12) + 'px';
  tip.style.top  = (evt.clientY - outer.top - 12) + 'px';
  tip.style.display = 'block';
}

function hideChartTooltip() {
  const tip = document.getElementById('fc-chart-tooltip');
  if (tip) tip.style.display = 'none';
}

// ── Utility ───────────────────────────────────────────────────────────────────

function statCard(label, value, sub) {
  const d = document.createElement('div');
  d.className = 'stat-card';
  d.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-sub">${sub}</div>`;
  return d;
}

function listRow(label, value) {
  const d = document.createElement('div');
  d.className = 'list-row';
  d.innerHTML = `<span class="list-label">${label}</span><span class="list-value">${value}</span>`;
  return d;
}

function fmtINR(v) {
  if (v === undefined || v === null) return '—';
  if (v >= 10000000) return '₹' + (v/10000000).toFixed(2) + ' Cr';
  if (v >= 100000)   return '₹' + (v/100000).toFixed(2) + ' L';
  return '₹' + Number(v).toLocaleString('en-IN', { maximumFractionDigits: 0 });
}

function fmtNum(v) {
  return Number(v).toLocaleString('en-IN');
}

function fmtShort(v) {
  if (v >= 10000000) return (v/10000000).toFixed(1)+'Cr';
  if (v >= 100000)   return (v/100000).toFixed(1)+'L';
  if (v >= 1000)     return (v/1000).toFixed(0)+'K';
  return Math.round(v);
}

// ── Forecast Analysis (Forecast Intelligence Engine — Phase 1) ────────────
// Dataset analysis, series classification, and candidate model
// recommendations for the Forecast page. Advisory only — does not affect
// the existing forecast chart/cards or /api/forecast in any way.

async function loadForecastAnalysis() {
  try {
    const res  = await fetch('/api/forecast/analysis');
    const data = await res.json();
    if (!data.ok) return;
    renderForecastAnalysis(data);
  } catch (e) {
    // Non-fatal — existing forecast functionality must keep working even
    // if the analysis endpoint is unavailable for any reason.
    console.warn('Forecast analysis unavailable:', e);
  }
}

function renderForecastAnalysis(data) {
  // Redesign: this data now lives inside the collapsed-by-default
  // "Dataset Analysis" accordion (Section 5) instead of sitting in the
  // main flow — same fetch, same data, just demoted from primary to
  // progressive-disclosure per the new information hierarchy. Populated
  // in the background as soon as it arrives so it's ready the instant a
  // user expands the accordion.
  document.getElementById('forecast-analysis-empty').style.display   = 'none';
  document.getElementById('forecast-analysis-content').style.display = 'block';

  const stats = data.stats;
  const cls   = data.classification;
  const recs  = data.recommendations;

  document.getElementById('fa-series-badge').textContent = cls.series_type;
  const confPill = document.getElementById('fa-confidence-pill');
  confPill.textContent = Math.round(cls.confidence * 100) + '%';
  confPill.className = 'conf-pill ' +
    (cls.confidence >= 0.75 ? 'conf-high' : cls.confidence >= 0.6 ? 'conf-medium' : 'conf-low');

  const grid = document.getElementById('fa-stats-grid');
  grid.innerHTML = '';
  [
    ['Observations',            fmtNum(stats.n_observations),                'data points'],
    ['Date Range',              `${stats.start_date} → ${stats.end_date}`,   ''],
    ['Frequency',               capitalize(stats.frequency),                 'inferred'],
    ['Missing Timestamps',      fmtNum(stats.missing_timestamps),            'gaps in series'],
    ['Missing Values',         fmtNum(stats.missing_values),                 'nulls remaining'],
    ['Variance',                fmtShort(stats.variance),                    ''],
    ['Coefficient of Variation', stats.coefficient_of_variation,             ''],
    ['Zero-Sales Periods',      stats.pct_zero_sales + '%',                  ''],
    ['Outliers',                stats.outlier_percentage + '%',              'of observations'],
    ['History Length',          fmtNum(stats.history_length_days) + ' days', ''],
  ].forEach(([label, value, sub]) => grid.appendChild(statCard(label, value, sub)));

  const modelsEl = document.getElementById('fa-models');
  modelsEl.innerHTML = '';
  recs.forEach((r, i) => {
    const row = document.createElement('div');
    row.className = 'list-row';
    row.innerHTML = `<span class="list-label">${r.rank}. ${r.model}</span>` +
                     `<span class="cat-pill">${i === 0 ? 'Recommended' : 'Alternative'}</span>`;
    modelsEl.appendChild(row);

    const desc = document.createElement('div');
    desc.className = 'anomaly-desc';
    desc.style.margin = '-4px 0 12px 0';
    desc.textContent = r.reason;
    modelsEl.appendChild(desc);
  });

  const reasonsEl = document.getElementById('fa-reasons');
  reasonsEl.innerHTML = '';
  cls.reasons.forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    reasonsEl.appendChild(li);
  });
}

function capitalize(s) {
  if (!s) return '—';
  return s.charAt(0).toUpperCase() + s.slice(1);
}


// ── Auto Forecasting Engine (Phase 2) ──────────────────────────────────────
// Trains every candidate model, evaluates them, and displays the
// automatically-selected best model alongside the full comparison table.
// Triggered explicitly by the "Run Auto Forecast" button (not on every
// page load) since training SARIMA/Holt-Winters/Prophet can be slow —
// this keeps the existing loadData() sequence exactly as fast as before.

// ── Phase 5: forecast level (segment) selection ────────────────────────────
// Every column and value below comes from /api/forecast/groups against the
// loaded dataset. Nothing is hard-coded, so an arbitrary uploaded ERP CSV
// gets whatever dimensions it actually has.
let groupMeta = null;

async function loadGroupMetadata() {
  try {
    const res = await fetch('/api/forecast/groups');
    const data = await res.json();
    if (!data.ok) return;
    groupMeta = data;

    const sel = document.getElementById('fc-group-col');
    if (!sel) return;
    sel.innerHTML = '<option value="">Company Total</option>';
    (data.group_columns || []).forEach(col => {
      const opt = document.createElement('option');
      opt.value = col.name;
      opt.textContent = col.label || col.name;
      sel.appendChild(opt);
    });
  } catch (err) {
    // Non-fatal: without metadata the selector simply stays at Company
    // Total and everything behaves as it did before Phase 5.
    console.warn('Group metadata unavailable:', err);
  }
}

function onGroupColChange() {
  const colSel = document.getElementById('fc-group-col');
  const valSel = document.getElementById('fc-group-value');
  const colName = colSel.value;

  if (!colName) {
    valSel.style.display = 'none';
    valSel.innerHTML = '';
    return;
  }

  const col = ((groupMeta && groupMeta.group_columns) || []).find(c => c.name === colName);
  valSel.innerHTML = '';
  ((col && col.values) || []).forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.value;
    // Flag segments whose history cannot support reliable validation, so
    // the user knows before spending a run on them.
    const short = v.history_status && v.history_status !== 'SUFFICIENT_HISTORY';
    opt.textContent = v.value + ` (${v.n_periods}p${short ? ' ⚠' : ''})`;
    valSel.appendChild(opt);
  });
  valSel.style.display = ((col && col.values) || []).length ? 'inline-block' : 'none';
}

// Shown when a segment cannot be resolved — unknown column, unknown
// value, or no matching transactions. Displays the backend's explanation
// instead of stale numbers from a previous run.
function renderGroupProblem(data) {
  const banner = document.getElementById('fc-scope-banner');
  banner.style.display = 'block';
  banner.classList.add('fc-scope-problem');
  document.getElementById('fc-scope-value').textContent = data.group_label || 'Unresolved segment';
  document.getElementById('fc-scope-meta').textContent =
    data.group_message || 'This segment could not be forecast.';
  document.getElementById('forecast-content').style.display = 'none';
  document.getElementById('forecast-empty').style.display = 'block';
}

function renderForecastScope(data) {
  const banner = document.getElementById('fc-scope-banner');
  if (!banner) return;
  banner.style.display = 'block';
  banner.classList.remove('fc-scope-problem');
  document.getElementById('fc-scope-value').textContent = data.group_label || 'Company Total';

  const bits = [];
  if (data.group_records != null && data.group_col) bits.push(`${data.group_records} transactions`);
  if (data.n_modelled_periods != null) bits.push(`${data.n_modelled_periods} ${data.frequency} periods`);
  if (data.group_history_status && data.group_history_status !== 'SUFFICIENT_HISTORY') {
    bits.push(data.group_history_status.replace(/_/g, ' ').toLowerCase());
  }
  document.getElementById('fc-scope-meta').textContent = bits.length ? '· ' + bits.join(' · ') : '';
}

async function runAutoForecast() {
  if (!loaded) {
    alert('Please load the ERP data first by clicking "Load ERP Data".');
    return;
  }
  const btn = document.getElementById('af-run-btn');
  const horizon = document.getElementById('forecast-horizon').value;
  btn.disabled = true;
  btn.textContent = 'Training models…';

  // Phase 5: send the selected segment, if any. Empty strings are sent as
  // nulls so an unselected level means "company total" — byte-identical
  // to the pre-Phase-5 request.
  const groupCol = (document.getElementById('fc-group-col') || {}).value || null;
  const groupValue = (document.getElementById('fc-group-value') || {}).value || null;

  try {
    const res = await fetch('/api/forecast/auto', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        months: parseInt(horizon),
        group_col: groupCol,
        group_value: groupCol ? groupValue : null,
      }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error);

    // A segment that could not be resolved comes back with a status and a
    // message rather than numbers. Showing the previous result here would
    // be exactly the company/segment mix-up Phase 5 exists to prevent.
    if (data.group_status && data.group_status !== 'OK') {
      renderGroupProblem(data);
      return;
    }
    autoForecastActive = true;
    renderAutoForecast(data);
  } catch (err) {
    alert('Auto forecast failed: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run Auto Forecast';
  }
}

// Redesign: this is now the ONLY writer of Section 1's hero cards,
// Section 2's chart, and Section 3's monthly cards once it has run —
// same DOM targets renderForecast() uses for the baseline view, so the
// page transitions from "simple preview" to "AI-selected result" in
// place rather than showing two competing forecasts at once.
// Phase 4 — render the prediction intervals, or an honest explanation of
// why none exists. Deliberately kept separate from the confidence pill
// above it: confidence is a reliability score for the model and the data,
// an interval is a range with a measured coverage level. Adding intervals
// never changes the confidence value.
function renderPredictionIntervals(data, forecastPts) {
  const block  = document.getElementById('fc-interval-block');
  const absent = document.getElementById('fc-interval-unavailable');
  const info   = data.prediction_intervals;

  if (!info || !info.available) {
    block.style.display = 'none';
    absent.style.display = 'block';
    absent.textContent = (info && info.unavailable_reason)
      || 'Prediction interval unavailable — insufficient validation history.';
    return;
  }

  absent.style.display = 'none';
  block.style.display = 'block';

  // Horizon totals: summing per-period bounds gives the range for the
  // whole window under the (stated) assumption that periods move
  // together. Labelled as a total so it is not mistaken for a per-period
  // range.
  const sum = (key) => forecastPts.reduce((t, p) => t + (p[key] != null ? p[key] : p.value), 0);
  document.getElementById('fc-interval-80').textContent =
    `${fmtINR(sum('lower_80'))} – ${fmtINR(sum('upper_80'))}`;
  document.getElementById('fc-interval-95').textContent =
    `${fmtINR(sum('lower_95'))} – ${fmtINR(sum('upper_95'))}`;

  const per = (k) => forecastPts.length === 1 ? 'for the forecast period'
    : `total across ${forecastPts.length} periods`;
  document.getElementById('fc-interval-80-sub').textContent = per('80');
  document.getElementById('fc-interval-95-sub').textContent = per('95');

  const status = info.interval_calibration_status || '—';
  const pill = document.getElementById('fc-interval-status');
  pill.textContent = status.replace(/_/g, ' ');
  pill.className = 'conf-pill-lg ' +
    (status === 'CALIBRATED' ? 'conf-high' : status === 'LIMITED_DATA' ? 'conf-medium' : 'conf-low');

  const cov80 = (info.coverage && info.coverage['80']) || {};
  const cov95 = (info.coverage && info.coverage['95']) || {};
  document.getElementById('fc-interval-status-sub').textContent =
    (cov80.observed != null)
      ? `measured coverage ${cov80.observed}% (80%) · ${cov95.observed}% (95%) on ${cov80.n} out-of-sample points`
      : 'coverage not yet measurable — too few validation folds';

  document.getElementById('fc-interval-note').textContent = (info.notes || []).join(' ');
}

// ── Phase 6: reliability factors, model explanation, insights ─────────────
// Every string rendered here comes from the API payload, which derives it
// from a measurement. Nothing is composed in the browser.
function renderExplanations(data) {
  const wrap = document.getElementById('fc-explain');
  if (!wrap) return;
  const rel = data.reliability;
  const model = data.model_explanation;
  const insights = data.forecast_insights;

  if (!rel && !model && !insights) {
    wrap.style.display = 'none';
    return;
  }
  wrap.style.display = 'block';

  // Why this reliability — one row per measured factor.
  const factorList = document.getElementById('fc-factor-list');
  factorList.innerHTML = '';
  ((rel && rel.factors) || []).forEach(f => {
    const li = document.createElement('li');
    li.className = 'fc-factor fc-factor-' + (f.impact || 'neutral');
    li.innerHTML =
      `<span class="fc-factor-label">${f.label}</span>` +
      `<span class="fc-factor-value">${f.value || '—'}</span>` +
      `<span class="fc-factor-why">${f.explanation}</span>`;
    factorList.appendChild(li);
  });

  const note = document.getElementById('fc-reliability-note');
  const warnings = ((rel && rel.warnings) || []);
  note.innerHTML =
    (rel ? `<div class="fc-explain-interp">${rel.interpretation}</div>` : '') +
    (warnings.length
      ? '<ul class="fc-warn-list">' + warnings.map(w => `<li>${w}</li>`).join('') + '</ul>'
      : '');

  // Why this model.
  const why = document.getElementById('fc-model-why');
  const altTable = document.getElementById('fc-alt-table');
  if (model) {
    why.innerHTML =
      `<p>${model.reason}</p>` +
      `<p class="fc-strength fc-strength-${(model.selection_strength || '').toLowerCase()}">` +
      `Selection strength: <strong>${model.selection_strength}</strong> — ${model.strength_reason}</p>`;
    const alts = (model.alternatives || []).slice(0, 6);
    altTable.innerHTML = alts.length
      ? '<table class="fc-alt"><thead><tr><th>Model</th><th>WAPE</th><th>Folds</th></tr></thead><tbody>'
        + alts.map(a =>
            `<tr class="${a.selected ? 'fc-alt-selected' : ''}">` +
            `<td>${a.model}${a.selected ? ' ✓' : ''}</td>` +
            `<td>${a.wape.toFixed(2)}%</td><td>${a.folds}</td></tr>`).join('')
        + '</tbody></table>'
      : '';
  } else {
    why.textContent = 'No model explanation is available for this forecast.';
    altTable.innerHTML = '';
  }

  // Forecast insights.
  const list = document.getElementById('fc-insight-list');
  list.innerHTML = '';
  ((insights && insights.insights) || []).forEach(i => {
    const li = document.createElement('li');
    li.className = 'fc-insight fc-insight-' + i.kind;
    li.innerHTML = `<span class="fc-insight-headline">${i.headline}</span>` +
                   `<span class="fc-insight-detail">${i.detail}</span>`;
    list.appendChild(li);
  });
}

function renderAutoForecast(data) {
  document.getElementById('forecast-empty').style.display   = 'none';
  document.getElementById('forecast-content').style.display = 'block';

  // Phase 5: label what this payload describes, read from the payload
  // itself rather than from the selector, so the banner can never
  // disagree with the numbers underneath it.
  renderForecastScope(data);

  // ── Section 1: hero summary ─────────────────────────────────────────────
  document.getElementById('fc-hero-model').textContent = data.selected_model;
  document.getElementById('fc-hero-model-tag').textContent = 'AI-selected · lowest forecast error';

  document.getElementById('fc-hero-horizon').textContent =
    data.horizon + (data.horizon === 1 ? ' Month' : ' Months');

  // Phase 6: show the reliability LEVEL, not a percentage that reads as a
  // probability. The score is secondary detail; "69%" beside the word
  // confidence invited exactly the reading the system cannot support.
  const rel = data.reliability;
  const confEl = document.getElementById('fc-hero-confidence');
  const confSub = document.getElementById('fc-hero-confidence-sub');
  if (rel) {
    confEl.textContent = rel.label;
    confEl.className = 'conf-pill-lg ' +
      (rel.level === 'HIGH' ? 'conf-high' :
       rel.level === 'MODERATE' ? 'conf-medium' : 'conf-low');
    confSub.textContent = `reliability score ${rel.score_pct}/100 — not a probability`;
  } else {
    // Pre-Phase-6 payloads (or a failed explanation build) still render.
    const confPct = Math.round((data.forecast_confidence || 0) * 100);
    confEl.textContent = confPct + '%';
    confEl.className = 'conf-pill-lg ' +
      (data.forecast_confidence >= 0.75 ? 'conf-high' :
       data.forecast_confidence >= 0.5 ? 'conf-medium' : 'conf-low');
    confSub.textContent = 'reliability score';
  }
  renderExplanations(data);

  const forecastPts = data.forecast || [];
  const total = forecastPts.reduce((sum, p) => sum + p.value, 0);
  document.getElementById('fc-hero-revenue').textContent = fmtINR(total);
  document.getElementById('fc-hero-revenue-sub').textContent =
    `total across ${forecastPts.length} month${forecastPts.length === 1 ? '' : 's'}`;

  // ── Section 1b: prediction intervals (Phase 4) ─────────────────────────
  renderPredictionIntervals(data, forecastPts);

  // ── Section 2: chart — shaded band is the real 80% prediction interval
  // when one exists. It used to be Math.min(0.5, mape/100), a symmetric
  // plus/minus MAPE ribbon with no coverage level behind it (and a
  // hardcoded plus/minus 15% when metrics were missing), which invited the
  // user to read a fabricated range as a genuine one.
  const histEntries = (kpiData && kpiData.monthly_revenue) ? Object.entries(kpiData.monthly_revenue).slice(-6) : [];
  const hasIntervals = forecastPts.some(p => p.lower_80 != null && p.upper_80 != null);
  document.getElementById('fc-legend-band').textContent =
    hasIntervals ? '80% Prediction Range' : 'Forecast';
  renderCombinedChart('forecast-chart', histEntries,
    forecastPts.map(p => ({
      label: p.period, value: p.value,
      lower: p.lower_80 != null ? p.lower_80 : null,
      upper: p.upper_80 != null ? p.upper_80 : null,
    })),
    null);

  // ── Section 3: monthly result cards + detail table ──────────────────────
  renderMonthCards(forecastPts);

  // ── Section 4: Selected Model (concise) ─────────────────────────────────
  document.getElementById('fc-model-empty').style.display = 'none';
  document.getElementById('fc-model-content').style.display = 'block';
  document.getElementById('fc-model-name').textContent = data.selected_model;

  const metricsEl = document.getElementById('fc-model-metrics');
  metricsEl.innerHTML = '';
  if (data.evaluation_metrics) {
    const m = data.evaluation_metrics;
    [['MAE', fmtShort(m.mae)], ['RMSE', fmtShort(m.rmse)], ['MAPE', m.mape + '%'], ['SMAPE', m.smape + '%']]
      .forEach(([label, value]) => {
        const chip = document.createElement('span');
        chip.className = 'fc-metric-chip';
        chip.innerHTML = `<span class="fc-metric-chip-label">${label}</span><span class="fc-metric-chip-value">${value}</span>`;
        metricsEl.appendChild(chip);
      });
  } else {
    metricsEl.innerHTML = '<span class="fc-metric-chip-note">No validation metrics — limited history</span>';
  }

  const reasonsEl = document.getElementById('fc-model-reasons');
  reasonsEl.innerHTML = '';
  (data.selection_reason || []).forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    reasonsEl.appendChild(li);
  });

  const altEl = document.getElementById('fc-alt-list');
  altEl.innerHTML = '';
  if (data.alternative_models && data.alternative_models.length > 0) {
    data.alternative_models.forEach(m => {
      const pill = document.createElement('span');
      pill.className = 'cat-pill';
      pill.style.marginRight = '6px';
      pill.textContent = m;
      altEl.appendChild(pill);
    });
  } else {
    altEl.innerHTML = '<span class="fc-metric-chip-note">No other model trained successfully</span>';
  }

  // ── Section 6: Model Comparison (inside the collapsed accordion) ───────
  document.getElementById('fc-comparison-empty').style.display = 'none';
  const cmpTable = document.getElementById('af-comparison-table');
  cmpTable.style.display = 'table';
  const cmpBody = document.getElementById('af-comparison-tbody');
  cmpBody.innerHTML = '';
  (data.model_ranking || []).forEach(m => {
    const status = !m.available ? 'Unavailable'
                 : !m.trained   ? 'Not Evaluated'
                 : m.model === data.selected_model ? 'Selected'
                 : 'Evaluated';
    // Colour usage stays meaningful: green = selected (good), blue = a
    // model that trained fine but wasn't picked, neutral grey = not
    // applicable to this dataset. Nothing here is red — a skipped model
    // is not a failure of the system.
    const statusClass = status === 'Selected' ? 'conf-high' : status === 'Evaluated' ? 'conf-medium' : 'conf-neutral';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${m.rank !== null && m.rank !== undefined ? m.rank : '—'}</td>
      <td>${m.model}</td>
      <td><span class="conf-pill ${statusClass}">${status}</span></td>
      <td>${m.metrics ? fmtShort(m.metrics.mae)  : '—'}</td>
      <td>${m.metrics ? fmtShort(m.metrics.rmse) : '—'}</td>
      <td>${m.metrics ? m.metrics.mape + '%'  : '—'}</td>
      <td>${m.metrics ? m.metrics.smape + '%' : '—'}</td>
    `;
    cmpBody.appendChild(tr);
  });

  // ── Section 7: Warnings — business-friendly wording only ───────────────
  renderForecastWarnings(data.warnings || [], data.model_ranking || []);
}

// Translates backend warning/error strings into plain business language.
// The raw strings (exception text, dependency names, thresholds) are never
// shown in the UI — only here, pattern-matched into a friendly sentence.
// Anything unrecognised falls through to a generic, jargon-free message
// rather than leaking implementation detail.
function humanizeWarning(raw) {
  const modelMatch = raw.match(/^([A-Za-z][A-Za-z0-9 \-]*?)\s+(?:recommended|unavailable|needs|failed)/i);
  const model = modelMatch ? modelMatch[1].trim() : null;

  if (/not(?: yet)? implemented/i.test(raw)) {
    return model ? `${model} is not available in this environment yet.` : 'One recommended model is not available in this environment yet.';
  }
  if (/dependency is not installed/i.test(raw)) {
    return model ? `${model} was not evaluated because a required component is not installed.` : 'A recommended model was not evaluated because a required component is not installed.';
  }
  if (/needs at least .* observations/i.test(raw)) {
    return model ? `${model} was not evaluated because the dataset does not have enough history yet.` : 'A model was not evaluated because the dataset does not have enough history yet.';
  }
  if (/failed to train/i.test(raw)) {
    return model ? `${model} was not evaluated because the dataset did not meet its requirements.` : 'A model was not evaluated because the dataset did not meet its requirements.';
  }
  if (/insufficient historical data|too little history/i.test(raw)) {
    return "There isn't enough historical data yet for a full model comparison — showing the best available baseline.";
  }
  if (/no valid.*observations/i.test(raw)) {
    return 'No usable sales history was found for this dataset.';
  }
  // Generic fallback: drop anything after a colon (typically raw
  // exception/library detail) and soften the remaining wording.
  return raw.replace(/:\s*.+$/, '.').replace(/['"]/g, '');
}

function renderForecastWarnings(warnings, modelRanking) {
  const el = document.getElementById('fc-warnings');
  el.innerHTML = '';

  if (!warnings.length) {
    el.style.display = 'none';
    return;
  }

  const totalTrained = modelRanking.filter(m => m.trained).length;
  el.style.display = 'block';

  const seen = new Set();
  warnings.forEach(w => {
    const friendly = humanizeWarning(w);
    if (seen.has(friendly)) return;   // de-duplicate identical friendly messages
    seen.add(friendly);
    const item = document.createElement('div');
    // Severity stays meaningful: orange = informational (a model was
    // skipped, others still ran fine); red is reserved for the rare case
    // where nothing could be trained at all.
    item.className = 'fc-warning-item' + (totalTrained === 0 ? ' fc-warning-severe' : '');
    item.textContent = friendly;
    el.appendChild(item);
  });
}
