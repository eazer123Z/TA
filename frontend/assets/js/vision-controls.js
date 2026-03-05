async function refreshVisionStatus() {
  try {
    const data = await apiGet('/system/status');
    const el = document.getElementById('visionStatus');
    if (el) el.textContent = `Status: ${data.vision.running ? 'RUNNING' : 'STOPPED'} | model=${data.vision.model}`;
  } catch (err) {
    const el = document.getElementById('visionStatus');
    if (el) el.textContent = `Status error: ${err.message}`;
  }
}

async function startVision() {
  await apiPost('/system/start');
  await refreshVisionStatus();
}

async function stopVision() {
  await apiPost('/system/stop');
  await refreshVisionStatus();
}
