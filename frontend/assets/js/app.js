document.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('btnStartVision')?.addEventListener('click', startVision);
  document.getElementById('btnStopVision')?.addEventListener('click', stopVision);

  document.getElementById('btnSaveConfig')?.addEventListener('click', async () => {
    const mqtt_host = document.getElementById('mqttHost')?.value || '';
    const mqtt_port = parseInt(document.getElementById('mqttPort')?.value || '1883', 10);
    await apiPost('/config', { mqtt_host, mqtt_port });
    await refreshVisionStatus();
  });

  try {
    const cfg = await apiGet('/config');
    document.getElementById('mqttHost').value = cfg.mqtt_host;
    document.getElementById('mqttPort').value = cfg.mqtt_port;
  } catch (_) {}

  await refreshVisionStatus();
  await loadActivitySummary();
});
