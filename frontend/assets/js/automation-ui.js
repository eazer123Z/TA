async function loadActivitySummary() {
  const ul = document.getElementById('activityList');
  if (!ul) return;
  ul.innerHTML = '';
  try {
    const rows = await apiGet('/activity/summary');
    rows.forEach((row) => {
      const li = document.createElement('li');
      li.textContent = `${row.event_type} • ${row.summary} • count=${row.count}`;
      ul.appendChild(li);
    });
  } catch (err) {
    const li = document.createElement('li');
    li.textContent = `Gagal memuat aktivitas: ${err.message}`;
    ul.appendChild(li);
  }
}
