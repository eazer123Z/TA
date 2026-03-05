const CONFIG = {
  mqtt: { broker: 'broker.hivemq.com', port: 8884, path: '/mqtt', maxReconnect: 5, reconnectDelay: 2000 },
  app: { maxLogs: 300 }
};

const STATE = {
  devices: {}, deviceStates: {}, deviceTopics: {}, deviceOnAt: {},
  sensors: {}, sensorData: {}, sensorHistory: {},
  automationRules: {},
  logs: [], logFilter: '',
  quickControlDevices: [],
  mqtt: { client: null, connected: false, reconnectAttempts: 0 },
  camera: { stream: null, active: false, selectedDeviceId: null, availableDevices: [] },
  cv: { personCount: 0, personPresent: false, brightness: 0, lightCondition: 'unknown' },
  sessionStart: Date.now()
};

const CV = {
  modelLoaded: false, modelLoading: false, detecting: false,
  frameCount: 0, fps: 0, fpsTimer: null,
  confidence: 0.65
};

const PAGE_TITLES = {
  dashboard: 'Overview', devices: 'Perangkat', sensors: 'Sensor',
  automation: 'Aturan Otomasi', camera: 'Kamera & CV', analytics: 'Log Aktivitas', settings: 'Pengaturan'
};

const SENSOR_META = {
  temperature: { min: 20, max: 35 },
  humidity: { min: 0, max: 100 },
  air_quality: { min: 0, max: 500 },
  brightness: { min: 0, max: 1 },
  smoke: { min: 0, max: 100 },
  gas: { min: 0, max: 100 }
};

function escHtml(s){
  if (s == null) return '';
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function showToast(msg, type='info'){
  const cont = document.getElementById('toastContainer');
  if (!cont) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  cont.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

async function apiPost(action, data={}) {
  try {
    const res = await fetch(`api/handler.php?action=${encodeURIComponent(action)}`, {
      method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify(data)
    });
    return await res.json();
  } catch (e) {
    showToast('API error: ' + action, 'error');
    return null;
  }
}

function switchPage(page, el){
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  document.getElementById(`view-${page}`)?.classList.remove('hidden');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  el?.classList.add('active');
  const t = document.getElementById('pageTitle');
  if (t) t.textContent = PAGE_TITLES[page] || page;
  if (page === 'automation') renderAutomationView();
}

function initClock(){
  const tick = () => {
    const c = document.getElementById('clock');
    if (c) c.textContent = new Date().toLocaleTimeString('id-ID');
  };
  tick();
  setInterval(tick, 1000);
}

function loadFromPHP(){
  if (Array.isArray(PHP_DEVICES)) {
    PHP_DEVICES.forEach(d => {
      const id = String(d.id);
      STATE.devices[id] = { ...d, id };
      STATE.deviceStates[id] = false;
      STATE.deviceTopics[id] = { sub: d.topic_sub || '', pub: d.topic_pub || '' };
    });
  }
  if (Array.isArray(PHP_SENSORS)) {
    PHP_SENSORS.forEach(s => {
      const id = String(s.id);
      STATE.sensors[id] = { ...s, id };
      STATE.sensorData[id] = null;
      STATE.sensorHistory[id] = [];
    });
  }
  if (PHP_SETTINGS?.quick_control_devices) {
    try {
      const q = PHP_SETTINGS.quick_control_devices;
      STATE.quickControlDevices = Array.isArray(q) ? q.map(String) : JSON.parse(q).map(String);
    } catch (_) {}
  }
}

function updateDashboardStats(){
  const set = (id, v) => { const e=document.getElementById(id); if(e) e.textContent=v; };
  set('statActiveDevicesVal', Object.values(STATE.deviceStates).filter(Boolean).length);
  set('statSensorsOnlineVal', Object.values(STATE.sensorData).filter(v => v !== null && v !== undefined).length);
  set('statPersonCount', STATE.cv.personCount);
  set('navDeviceCount', Object.keys(STATE.devices).length);
  set('navSensorCount', Object.keys(STATE.sensors).length);
}

function renderDevices(){
  const grid = document.getElementById('devicesGrid');
  if (!grid) return;
  grid.innerHTML = '';
  Object.keys(STATE.devices).forEach(id => {
    const d = STATE.devices[id];
    const on = !!STATE.deviceStates[id];
    const card = document.createElement('div');
    card.className = 'device-card' + (on ? ' on' : '');
    card.id = `card-${id}`;
    card.innerHTML = `
      <div><b>${escHtml(d.name)}</b></div>
      <div>Status: ${on ? 'ON' : 'OFF'}</div>
      <div style="display:flex;gap:6px;margin-top:6px">
        <button onclick="toggleDeviceState('${id}', ${!on})">${on ? 'OFF' : 'ON'}</button>
        <button onclick="removeDevice('${id}')">Hapus</button>
      </div>`;
    grid.appendChild(card);
  });
}

function renderSensors(){
  const grid = document.getElementById('sensorsGrid');
  if (!grid) return;
  grid.innerHTML = '';
  Object.keys(STATE.sensors).forEach(id => {
    const s = STATE.sensors[id];
    const val = STATE.sensorData[id];
    const card = document.createElement('div');
    card.className = 'sensor-card';
    card.innerHTML = `
      <div><b>${escHtml(s.name)}</b></div>
      <div>${escHtml(s.type || '')}</div>
      <div>${val == null ? '—' : val}${escHtml(s.unit || '')}</div>
      <div style="display:flex;gap:6px;margin-top:6px">
        <button onclick="removeSensor('${id}')">Hapus</button>
      </div>`;
    grid.appendChild(card);
  });
}

function renderQuickControls(){
  const c = document.getElementById('quickControlsContainer');
  if (!c) return;
  c.innerHTML = '';
  const ids = STATE.quickControlDevices.length ? STATE.quickControlDevices : Object.keys(STATE.devices).slice(0, 6);
  ids.forEach(id => {
    const d = STATE.devices[String(id)];
    if (!d) return;
    const on = !!STATE.deviceStates[String(id)];
    const row = document.createElement('div');
    row.className = 'qc-item' + (on ? ' on' : '');
    row.innerHTML = `${escHtml(d.name)} <button onclick="toggleDeviceState('${id}', ${!on})">${on ? 'OFF' : 'ON'}</button>`;
    c.appendChild(row);
  });
}

function toggleDeviceState(deviceId, newState){
  const id = String(deviceId);
  STATE.deviceStates[id] = !!newState;
  if (newState) STATE.deviceOnAt[id] = Date.now(); else delete STATE.deviceOnAt[id];
  const t = STATE.deviceTopics[id];
  if (t?.pub && STATE.mqtt.client && STATE.mqtt.connected) {
    try {
      const m = new Paho.MQTT.Message(JSON.stringify({ state: newState ? 1 : 0 }));
      m.destinationName = t.pub;
      STATE.mqtt.client.send(m);
    } catch (_) {}
  }
  addLog(STATE.devices[id]?.name || 'Device', `Status ${newState ? 'ON' : 'OFF'}`, 'Manual', 'info');
  renderDevices();
  renderQuickControls();
  updateDashboardStats();
}

function applyDeviceState(deviceId, newState, reason='Automation'){
  if (STATE.deviceStates[String(deviceId)] === !!newState) return;
  toggleDeviceState(String(deviceId), !!newState);
  addLog(STATE.devices[String(deviceId)]?.name || 'Device', `Auto ${newState?'ON':'OFF'} (${reason})`, 'Automation', 'success');
}

function shouldFireRule(rule, val){
  switch (rule.condition) {
    case 'gt': return Number(val) > Number(rule.threshold);
    case 'lt': return Number(val) < Number(rule.threshold);
    case 'detected': return !!val;
    case 'absent': return !val;
    case 'range': return Number(val) < Number(rule.thresholdMin) || Number(val) > Number(rule.thresholdMax);
    default: return false;
  }
}

function checkAutomationRules(){
  Object.keys(STATE.sensors).forEach(sensorId => {
    const val = STATE.sensorData[sensorId];
    if (val == null) return;
    const rules = STATE.automationRules[sensorId] || [];
    rules.forEach(r => {
      if (!r.enabled) return;
      if (shouldFireRule(r, val)) {
        applyDeviceState(String(r.deviceId), r.action === 'on', `Auto(${STATE.sensors[sensorId]?.name || sensorId})`);
      }
    });
  });
}

function renderAutomationView(){
  const grid = document.getElementById('automationGrid');
  if (!grid) return;
  grid.innerHTML = '';
  Object.keys(STATE.sensors).forEach(sensorId => {
    const sensor = STATE.sensors[sensorId];
    const rules = STATE.automationRules[sensorId] || [];
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<b>${escHtml(sensor.name)}</b><div>${rules.length} rule(s)</div><button onclick="openAddRulePrompt('${sensorId}')">+ Rule</button>`;
    grid.appendChild(card);
  });
}

function openAddRulePrompt(sensorId){
  const dIds = Object.keys(STATE.devices);
  if (!dIds.length) { showToast('Tambahkan perangkat dulu', 'warning'); return; }
  const threshold = prompt('Threshold angka (contoh: 30):', '30');
  if (threshold == null) return;
  const rule = {
    ruleId: `r_${Date.now()}`,
    enabled: true,
    condition: 'gt',
    threshold: Number(threshold),
    deviceId: dIds[0],
    action: 'on'
  };
  if (!STATE.automationRules[String(sensorId)]) STATE.automationRules[String(sensorId)] = [];
  STATE.automationRules[String(sensorId)].push(rule);
  renderAutomationView();
  showToast('Rule ditambahkan', 'success');
}

function processSensorValue(sensorId, val){
  const id = String(sensorId);
  STATE.sensorData[id] = val;
  if (!STATE.sensorHistory[id]) STATE.sensorHistory[id] = [];
  STATE.sensorHistory[id].push(val);
  if (STATE.sensorHistory[id].length > 30) STATE.sensorHistory[id].shift();
  renderSensors();
  updateDashboardStats();
  checkAutomationRules();
  if (automationEngine?.isActive) automationEngine.evaluateSensorRules(id, val);
}

async function connectMQTT(){
  const broker = document.getElementById('mqttBroker')?.value || PHP_SETTINGS?.mqtt_broker || CONFIG.mqtt.broker;
  const port = parseInt(document.getElementById('mqttPort')?.value || PHP_SETTINGS?.mqtt_port || CONFIG.mqtt.port, 10);
  const clientId = `iotzy_web_${Math.random().toString(16).slice(2,8)}`;
  try {
    STATE.mqtt.client = new Paho.MQTT.Client(broker, port, '/mqtt', clientId);
    STATE.mqtt.client.onConnectionLost = () => { STATE.mqtt.connected = false; showToast('MQTT putus', 'warning'); };
    STATE.mqtt.client.onMessageArrived = msg => handleMQTTMessage(msg.destinationName, msg.payloadString);
    STATE.mqtt.client.connect({
      useSSL: true,
      onSuccess: () => {
        STATE.mqtt.connected = true;
        Object.values(STATE.deviceTopics).forEach(t => { if (t.sub) STATE.mqtt.client.subscribe(t.sub); });
        Object.values(STATE.sensors).forEach(s => { if (s.topic) STATE.mqtt.client.subscribe(s.topic); });
        showToast('MQTT connected', 'success');
      },
      onFailure: () => showToast('MQTT gagal', 'error')
    });
  } catch (_) {
    showToast('MQTT error', 'error');
  }
}

function handleMQTTMessage(topic, payload){
  for (const [id, t] of Object.entries(STATE.deviceTopics)) {
    if (t.sub === topic) {
      try {
        const data = JSON.parse(payload);
        STATE.deviceStates[id] = data.state === 1 || data.state === true || data.state === 'on';
      } catch (_) {
        const raw = String(payload).toLowerCase().trim();
        STATE.deviceStates[id] = raw === '1' || raw === 'on';
      }
      renderDevices();
      renderQuickControls();
      updateDashboardStats();
      return;
    }
  }
  for (const [id, s] of Object.entries(STATE.sensors)) {
    if (s.topic === topic) {
      const n = Number(payload);
      if (!Number.isNaN(n)) processSensorValue(id, n);
      return;
    }
  }
}

async function toggleCamera(){
  try {
    if (!STATE.camera.stream) {
      STATE.camera.stream = await navigator.mediaDevices.getUserMedia({ video: true });
      STATE.camera.active = true;
      document.getElementById('cameraFocus').srcObject = STATE.camera.stream;
      showToast('Kamera aktif', 'success');
    } else {
      stopCVDetection();
      STATE.camera.stream.getTracks().forEach(t => t.stop());
      STATE.camera.stream = null;
      STATE.camera.active = false;
      showToast('Kamera mati', 'info');
    }
  } catch (e) {
    showToast('Camera error: ' + e.message, 'error');
  }
}

async function initializeCV(){
  if (CV.modelLoading || CV.modelLoaded) return;
  CV.modelLoading = true;
  document.getElementById('cvSystemStatus').textContent = 'Loading model...';
  const ok = await cvDetector.initialize();
  CV.modelLoading = false;
  CV.modelLoaded = ok;
  document.getElementById('btnStartCV').disabled = !ok;
  document.getElementById('cvSystemStatus').textContent = ok ? 'Ready' : 'Error';
  if (ok) {
    cvUI.initialize();
    cvUI.renderAutomationSettings();
    showToast('Model COCO-SSD siap', 'success');
  }
}

function startCVDetection(){
  if (!CV.modelLoaded) return showToast('Load model dulu', 'warning');
  if (!STATE.camera.stream) return showToast('Aktifkan kamera dulu', 'warning');
  if (CV.detecting) return;
  CV.detecting = true;
  document.getElementById('btnStartCV').disabled = true;
  document.getElementById('btnStopCV').disabled = false;
  document.getElementById('cvDetectionInfo').style.display = 'flex';
  CV.frameCount = 0;
  CV.fpsTimer = setInterval(() => {
    CV.fps = CV.frameCount;
    CV.frameCount = 0;
    const e = document.getElementById('cvFpsStatus');
    if (e) e.textContent = `${CV.fps} fps`;
  }, 1000);
  cvDetector.startDetection(document.getElementById('cameraFocus'));
  lightAnalyzer.startAnalysis(document.getElementById('cameraFocus'));
  automationEngine.start();
}

function stopCVDetection(){
  if (!CV.detecting) return;
  CV.detecting = false;
  cvDetector.stopDetection();
  lightAnalyzer.stopAnalysis();
  automationEngine.stop();
  if (CV.fpsTimer) clearInterval(CV.fpsTimer);
  CV.fpsTimer = null;
  document.getElementById('btnStartCV').disabled = false;
  document.getElementById('btnStopCV').disabled = true;
  document.getElementById('cvDetectionInfo').style.display = 'none';
  cvUI.clearOverlay();
}

function onCVPersonCountUpdate(count){
  STATE.cv.personCount = count;
  STATE.cv.personPresent = count > 0;
  CV.frameCount++;
  const e = document.getElementById('cvHumanCount');
  if (e) e.textContent = String(count);
  updateDashboardStats();
}

async function addLog(device, activity, trigger, type='info'){
  const now = new Date();
  STATE.logs.unshift({
    tanggal: now.toLocaleDateString('id-ID'),
    waktu: now.toLocaleTimeString('id-ID'),
    device, activity, trigger, type
  });
  if (STATE.logs.length > CONFIG.app.maxLogs) STATE.logs.length = CONFIG.app.maxLogs;
  updateLogDisplay();
  await apiPost('add_log', { device, activity, trigger, type });
}

function updateLogDisplay(){
  const tbody = document.getElementById('logBody');
  if (!tbody) return;
  const q = STATE.logFilter.toLowerCase();
  tbody.innerHTML = '';
  STATE.logs
    .filter(l => (l.device || '').toLowerCase().includes(q) || (l.activity || '').toLowerCase().includes(q) || (l.trigger || '').toLowerCase().includes(q))
    .forEach(l => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${escHtml(l.tanggal || '')}</td><td>${escHtml(l.waktu || '')}</td><td>${escHtml(l.device || '')}</td><td>${escHtml(l.activity || '')}</td><td>${escHtml(l.trigger || '')}</td>`;
      tbody.appendChild(tr);
    });
}

function filterLogs(q){ STATE.logFilter = q || ''; updateLogDisplay(); }

function openAddDeviceModal(){ document.getElementById('addDeviceModal')?.classList.remove('hidden'); }
function closeAddDeviceModal(){ document.getElementById('addDeviceModal')?.classList.add('hidden'); }
function openAddSensorModal(){ document.getElementById('addSensorModal')?.classList.remove('hidden'); }
function closeAddSensorModal(){ document.getElementById('addSensorModal')?.classList.add('hidden'); }

async function saveNewDevice(){
  const name = document.getElementById('newDeviceName')?.value.trim();
  const sub = document.getElementById('newDeviceTopicSub')?.value.trim();
  const pub = document.getElementById('newDeviceTopicPub')?.value.trim();
  if (!name) return showToast('Nama perangkat wajib', 'warning');
  const r = await apiPost('add_device', { name, icon: 'fa-plug', topic_sub: sub, topic_pub: pub });
  if (r?.success) {
    const id = String(r.id);
    STATE.devices[id] = { id, name, icon: 'fa-plug', topic_sub: sub, topic_pub: pub };
    STATE.deviceStates[id] = false;
    STATE.deviceTopics[id] = { sub, pub };
    renderDevices(); renderQuickControls(); updateDashboardStats(); closeAddDeviceModal();
  }
}

async function removeDevice(deviceId){
  if (!confirm('Hapus perangkat ini?')) return;
  const r = await apiPost('delete_device', { id: String(deviceId) });
  if (r?.success) {
    delete STATE.devices[String(deviceId)];
    delete STATE.deviceStates[String(deviceId)];
    delete STATE.deviceTopics[String(deviceId)];
    renderDevices(); renderQuickControls(); updateDashboardStats();
  }
}

async function saveNewSensor(){
  const name = document.getElementById('newSensorName')?.value.trim();
  const type = document.getElementById('newSensorType')?.value || 'temperature';
  const topic = document.getElementById('newSensorTopic')?.value.trim();
  const unit = document.getElementById('newSensorUnit')?.value.trim();
  if (!name || !topic) return showToast('Nama/topic sensor wajib', 'warning');
  const r = await apiPost('add_sensor', { name, type, topic, unit });
  if (r?.success) {
    const id = String(r.id);
    STATE.sensors[id] = { id, name, type, topic, unit };
    STATE.sensorData[id] = null;
    STATE.sensorHistory[id] = [];
    renderSensors(); updateDashboardStats(); closeAddSensorModal();
  }
}

async function removeSensor(sensorId){
  if (!confirm('Hapus sensor ini?')) return;
  const r = await apiPost('delete_sensor', { id: String(sensorId) });
  if (r?.success) {
    delete STATE.sensors[String(sensorId)];
    delete STATE.sensorData[String(sensorId)];
    delete STATE.sensorHistory[String(sensorId)];
    delete STATE.automationRules[String(sensorId)];
    renderSensors(); renderAutomationView(); updateDashboardStats();
  }
}

async function saveMQTTConfig(){
  const mqtt_broker = document.getElementById('mqttBroker')?.value.trim();
  const mqtt_port = parseInt(document.getElementById('mqttPort')?.value || '8884', 10);
  const r = await apiPost('save_settings', { mqtt_broker, mqtt_port });
  if (r?.success) showToast('Config tersimpan', 'success');
}

document.addEventListener('DOMContentLoaded', async () => {
  loadCVConfig();
  loadFromPHP();
  initClock();
  renderDevices();
  renderSensors();
  renderQuickControls();
  renderAutomationView();
  updateDashboardStats();
  const logs = await apiPost('get_logs', {});
  if (Array.isArray(logs)) {
    STATE.logs = logs.map(r => ({
      tanggal: r.tanggal || '', waktu: r.waktu || '',
      device: r.device_name, activity: r.activity, trigger: r.trigger_type, type: r.log_type
    }));
    updateLogDisplay();
  }
});
