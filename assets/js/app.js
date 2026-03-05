// ============================================================
// IoTzy Dashboard v4 — app.js (Fixed & Enhanced)
// ============================================================

const CONFIG = {
    mqtt: { broker: 'broker.hivemq.com', port: 8884, path: '/mqtt', maxReconnect: 5, reconnectDelay: 3000 },
    app:  { maxLogs: 200, updateInterval: 1000 }
};

const STATE = {
    devices:             {},
    deviceStates:        {},
    deviceTopics:        {},
    deviceOnAt:          {},
    sensors:             {},
    sensorData:          {},
    sensorHistory:       {},
    automationRules:     {},
    logs:                [],
    logFilter:           '',
    logTypeFilter:       'all',
    quickControlDevices: [],
    mqtt: { client: null, connected: false, reconnectAttempts: 0 },
    camera: { stream: null, active: false, selectedDeviceId: null, availableDevices: [] },
    sessionStart: Date.now(),
    // CV runtime state for person counting
    cv: {
        personCount:    0,
        personPresent:  false,
        lightCondition: 'unknown',
        brightness:     0
    }
};

const CV = {
    modelLoaded:   false,
    modelLoading:  false,
    detecting:     false,
    model:         null,
    fpsTimer:      null,
    fps:           0,
    frameCount:    0,
    overlayCanvas: null,
    overlayCtx:    null,
    showBoxes:     true,
    showDebug:     true,
    confidence:    0.60,
    humanPresent:  false,
    humanTimer:    null,
    lightCondition:'unknown',
    lightTimer:    null,
    cvRules: {
        human: { enabled: true, onDetect: [], onAbsent: [], delay: 3000 },
        light: { enabled: true, onDark:   [], onBright: [], delay: 2000 }
    }
};

/* ==================== API ==================== */
async function apiPost(action, data = {}) {
    try {
        const res = await fetch(`api/handler.php?action=${action}`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(data)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (e) {
        console.error('API error:', action, e);
        showToast(`Error: ${action}`, 'error');
        return null;
    }
}

/* ==================== INIT ==================== */
document.addEventListener('DOMContentLoaded', async () => {
    initClock();
    loadCVConfig();
    loadFromPHP();
    initAutomationRules();
    automationEngine.initialize();
    loadCVRules();
    await loadLogs();
    renderAll();
    initUptimeCounter();

    setTimeout(() => {
        if (PHP_SETTINGS.mqtt_broker) connectMQTT();
    }, 900);

    setTimeout(() => {
        const ls  = document.getElementById('appLoadingScreen');
        const app = document.getElementById('mainApp');
        if (ls)  { ls.style.opacity = '0'; setTimeout(() => ls.style.display = 'none', 500); }
        if (app) app.classList.remove('opacity-0');
    }, 1300);
});

function loadFromPHP() {
    try {
        const qc = PHP_SETTINGS.quick_control_devices;
        STATE.quickControlDevices = Array.isArray(qc)
            ? qc.map(String)
            : (typeof qc === 'string' ? JSON.parse(qc || '[]').map(String) : []);
    } catch { STATE.quickControlDevices = []; }

    if (typeof PHP_DEVICES !== 'undefined') {
        PHP_DEVICES.forEach(d => {
            const id               = String(d.id);
            STATE.devices[id]      = { ...d, id };
            STATE.deviceStates[id] = false;
            STATE.deviceTopics[id] = { sub: d.topic_sub || '', pub: d.topic_pub || '' };
        });
    }

    if (typeof PHP_SENSORS !== 'undefined') {
        PHP_SENSORS.forEach(s => {
            const id                = String(s.id);
            STATE.sensors[id]       = { ...s, id };
            STATE.sensorData[id]    = null;
            STATE.sensorHistory[id] = [];
        });
    }
}

function renderAll() {
    renderDevices();
    renderSensors();
    renderQuickControls();
    updateDashboardStats();
}

/* ==================== CLOCK ==================== */
function initClock() {
    const tick = () => {
        const now = new Date();
        const el  = document.getElementById('clock');
        const de  = document.getElementById('date');
        if (el) el.textContent = now.toLocaleTimeString('id-ID');
        if (de) de.textContent = now.toLocaleDateString('id-ID', {
            weekday: 'short', day: 'numeric', month: 'short'
        }).toUpperCase();
    };
    tick();
    setInterval(tick, 1000);
}

function initUptimeCounter() {
    setInterval(() => {
        const e  = Math.floor((Date.now() - STATE.sessionStart) / 1000);
        const h  = Math.floor(e / 3600);
        const m  = Math.floor((e % 3600) / 60);
        const s  = e % 60;
        const el = document.getElementById('statUptimeVal');
        if (el) el.textContent =
            `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
        updateAllDurations();
    }, 1000);
}

function updateAllDurations() {
    Object.keys(STATE.devices).forEach(id => {
        const el = document.getElementById(`dur-${id}`);
        if (!el) return;
        if (STATE.deviceStates[id] && STATE.deviceOnAt[id]) {
            const sec = Math.floor((Date.now() - STATE.deviceOnAt[id]) / 1000);
            const h   = Math.floor(sec / 3600);
            const m   = Math.floor((sec % 3600) / 60);
            const s   = sec % 60;
            el.textContent = h > 0 ? `${h}j ${m}m` : m > 0 ? `${m}m ${s}d` : `${s}d nyala`;
            el.className   = 'device-duration on';
        } else {
            el.textContent = 'Mati';
            el.className   = 'device-duration';
        }
    });
}

/* ==================== NAVIGATION ==================== */
const PAGE_TITLES = {
    dashboard:  'Overview',
    devices:    'Perangkat',
    sensors:    'Sensor',
    automation: 'Aturan Otomasi',
    camera:     'Kamera & CV',
    analytics:  'Log Aktivitas',
    settings:   'Pengaturan'
};

function switchPage(page, el) {
    document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
    document.getElementById(`view-${page}`)?.classList.remove('hidden');
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    if (el) el.classList.add('active');
    const pt = document.getElementById('pageTitle');
    if (pt) pt.textContent = PAGE_TITLES[page] || page;

    if (page === 'automation') renderAutomationView();
    if (page === 'camera') {
        setTimeout(() => {
            const c    = document.getElementById('cvOverlayCanvas');
            const cont = document.getElementById('cameraFocusContainer');
            if (c && cont) {
                c.width  = cont.clientWidth;
                c.height = cont.clientHeight;
            }
            if (typeof cvUI !== 'undefined' && CV.modelLoaded) {
                cvUI.renderAutomationSettings();
            }
        }, 100);
    }
}

function toggleSidebar() {
    document.getElementById('sidebar')?.classList.toggle('open');
    document.getElementById('overlay')?.classList.toggle('show');
}

/* ==================== STATS ==================== */
function updateDashboardStats() {
    const totalDev  = Object.keys(STATE.devices).length;
    const activeDev = Object.values(STATE.deviceStates).filter(Boolean).length;
    const totalSen  = Object.keys(STATE.sensors).length;
    const activeSen = Object.values(STATE.sensorData).filter(v => v !== null && v !== undefined).length;
    const g         = id => document.getElementById(id);

    if (g('statActiveDevicesVal')) g('statActiveDevicesVal').textContent = activeDev;
    if (g('statActiveDevicesSub')) g('statActiveDevicesSub').textContent = `dari ${totalDev} total`;
    if (g('statSensorsOnlineVal')) g('statSensorsOnlineVal').textContent = activeSen;
    if (g('statSensorsOnlineSub')) g('statSensorsOnlineSub').textContent = `dari ${totalSen} sensor`;
    if (g('navDeviceCount'))       g('navDeviceCount').textContent       = totalDev;
    if (g('navSensorCount'))       g('navSensorCount').textContent       = totalSen;
    if (g('totalDevices'))         g('totalDevices').textContent         = totalDev;
    if (g('totalSensors'))         g('totalSensors').textContent         = totalSen;
    if (g('statMqttVal'))          g('statMqttVal').textContent          = STATE.mqtt.connected ? 'Online' : 'Offline';
    if (g('statMqttSub'))          g('statMqttSub').textContent          = PHP_SETTINGS.mqtt_broker || '—';

    // Person count stats
    const pc = STATE.cv.personCount;
    if (g('statPersonCount'))    g('statPersonCount').textContent    = pc;
    if (g('statPersonPresence')) g('statPersonPresence').textContent = pc > 0 ? `${pc} orang terdeteksi` : 'Tidak ada';
    if (g('cvPersonCountBig'))   g('cvPersonCountBig').textContent   = pc;
}

/* ==================== CAMERA ==================== */
async function listCameraDevices() {
    try {
        const devs = await navigator.mediaDevices.enumerateDevices();
        STATE.camera.availableDevices = devs.filter(d => d.kind === 'videoinput');
        return STATE.camera.availableDevices;
    } catch { return []; }
}

function openCameraSelector() {
    document.getElementById('cameraSelectorModal')?.classList.add('show');
    const list = document.getElementById('cameraDevicesList');
    if (!list) return;
    listCameraDevices().then(devs => {
        list.innerHTML = '';
        if (!devs.length) {
            list.innerHTML = '<p class="modal-loading">Tidak ada kamera ditemukan</p>';
            return;
        }
        devs.forEach((dev, idx) => {
            const btn     = document.createElement('button');
            btn.className = 'modal-item' + (dev.deviceId === STATE.camera.selectedDeviceId ? ' selected' : '');
            btn.innerHTML = `<i class="fas fa-camera"></i> ${dev.label || 'Kamera ' + (idx + 1)}`;
            btn.onclick   = () => { selectCamera(dev.deviceId); closeCameraSelector(); };
            list.appendChild(btn);
        });
    });
}

function closeCameraSelector() {
    document.getElementById('cameraSelectorModal')?.classList.remove('show');
}

async function selectCamera(deviceId) {
    STATE.camera.selectedDeviceId = deviceId;
    if (STATE.camera.stream) {
        STATE.camera.stream.getTracks().forEach(t => t.stop());
        STATE.camera.stream = null;
        await startCamera();
    } else {
        showToast('Kamera dipilih, klik power untuk aktifkan', 'info');
    }
}

async function startCamera() {
    const constraints = STATE.camera.selectedDeviceId
        ? { video: { deviceId: { exact: STATE.camera.selectedDeviceId } } }
        : { video: true };
    STATE.camera.stream = await navigator.mediaDevices.getUserMedia(constraints);
    STATE.camera.active = true;
    updateCameraElements(true);
}

async function toggleCamera() {
    try {
        if (!STATE.camera.stream) {
            await startCamera();
            showToast('Kamera aktif', 'success');
        } else {
            if (CV.detecting) stopCVDetection();
            STATE.camera.stream.getTracks().forEach(t => t.stop());
            STATE.camera.stream = null;
            STATE.camera.active = false;
            updateCameraElements(false);
            showToast('Kamera dimatikan', 'info');
        }
    } catch (e) {
        showToast('Tidak dapat mengakses kamera: ' + e.message, 'error');
    }
}

function toggleCameraFocus() { toggleCamera(); }

function updateCameraElements(isActive) {
    [
        { v: 'camera',      p: 'camPlaceholder',         b: 'camTag' },
        { v: 'cameraFocus', p: 'cameraFocusPlaceholder', b: 'cameraFocusTag' }
    ].forEach(({ v, p, b }) => {
        const vid = document.getElementById(v);
        const ph  = document.getElementById(p);
        const tag = document.getElementById(b);
        if (!vid || !ph) return;
        if (isActive && STATE.camera.stream) {
            vid.srcObject = STATE.camera.stream;
            vid.classList.remove('hidden');
            ph.style.display = 'none';
            if (tag) tag.classList.remove('hidden');
        } else {
            vid.srcObject    = null;
            vid.classList.add('hidden');
            ph.style.display = null;
            if (tag) tag.classList.add('hidden');
        }
    });
}

/* ==================== MQTT ==================== */
async function connectMQTT() {
    try {
        const broker   = document.getElementById('mqttBroker')?.value    || PHP_SETTINGS.mqtt_broker   || CONFIG.mqtt.broker;
        const port     = parseInt(document.getElementById('mqttPort')?.value)   || PHP_SETTINGS.mqtt_port   || CONFIG.mqtt.port;
        const clientId = (document.getElementById('mqttClientId')?.value  || 'iotzy_web')
                         + '_' + Math.random().toString(16).substr(2, 6);
        const path     = document.getElementById('mqttPath')?.value       || PHP_SETTINGS.mqtt_path     || CONFIG.mqtt.path;
        const useSSL   = document.getElementById('mqttUseSSL')?.checked   ?? !!PHP_SETTINGS.mqtt_use_ssl;
        const user     = document.getElementById('mqttUsername')?.value   || PHP_SETTINGS.mqtt_username || '';
        const pass     = document.getElementById('mqttPassword')?.value   || '';

        if (STATE.mqtt.client && STATE.mqtt.connected) {
            try { STATE.mqtt.client.disconnect(); } catch (_) {}
        }

        STATE.mqtt.client = new Paho.MQTT.Client(broker, port, path, clientId);

        STATE.mqtt.client.onConnectionLost = res => {
            updateMQTTStatus(false);
            addLog('MQTT', 'Terputus: ' + res.errorMessage, 'System', 'error');
            if (res.errorCode !== 0 && STATE.mqtt.reconnectAttempts < CONFIG.mqtt.maxReconnect) {
                const delay = CONFIG.mqtt.reconnectDelay * Math.pow(2, STATE.mqtt.reconnectAttempts++);
                setTimeout(() => { if (!STATE.mqtt.connected) connectMQTT(); }, delay);
            }
        };
        STATE.mqtt.client.onMessageArrived = msg =>
            handleMQTTMessage(msg.destinationName, msg.payloadString);

        const opts = {
            timeout: 10, keepAliveInterval: 30, cleanSession: true, useSSL,
            onSuccess: () => {
                STATE.mqtt.reconnectAttempts = 0;
                updateMQTTStatus(true);
                addLog('MQTT', `Terhubung ke ${broker}`, 'System', 'success');
                showToast('MQTT terhubung!', 'success');
                subscribeToAllTopics();
                updateDashboardStats();
            },
            onFailure: err => {
                updateMQTTStatus(false);
                addLog('MQTT', 'Gagal: ' + (err.errorMessage || 'Unknown'), 'System', 'error');
                showToast('Koneksi MQTT gagal', 'error');
            }
        };
        if (user) opts.userName = user;
        if (pass) opts.password = pass;
        STATE.mqtt.client.connect(opts);
    } catch (e) {
        showToast('Error MQTT: ' + e.message, 'error');
    }
}

function disconnectMQTT() {
    if (STATE.mqtt.client && STATE.mqtt.connected) {
        try { STATE.mqtt.client.disconnect(); } catch (_) {}
    }
    updateMQTTStatus(false);
    showToast('MQTT diputus', 'info');
}

function updateMQTTStatus(connected) {
    STATE.mqtt.connected = connected;
    const g = id => document.getElementById(id);

    g('mqttStatusDot')?.classList.toggle('connected', connected);
    if (g('mqttStatusText')) g('mqttStatusText').textContent = connected ? 'Connected' : 'Disconnected';
    g('sidebarMqttDot')?.classList.toggle('connected', connected);
    if (g('sidebarMqttText')) g('sidebarMqttText').textContent = connected ? 'Online' : 'Offline';

    const sv = g('mqttStatusSettings');
    if (sv) {
        sv.textContent = connected ? 'Terhubung' : 'Disconnected';
        sv.className   = 'setting-val ' + (connected ? 'ok' : 'muted');
    }
    updateDashboardStats();
}

function subscribeToAllTopics() {
    if (!STATE.mqtt.connected || !STATE.mqtt.client) return;
    Object.values(STATE.deviceTopics).forEach(t => {
        if (t.sub) try { STATE.mqtt.client.subscribe(t.sub); } catch (_) {}
    });
    Object.values(STATE.sensors).forEach(s => {
        if (s.topic) try { STATE.mqtt.client.subscribe(s.topic); } catch (_) {}
    });
}

function publishMQTT(topic, payload) {
    if (!STATE.mqtt.connected || !STATE.mqtt.client || !topic) return false;
    try {
        const msg           = new Paho.MQTT.Message(JSON.stringify(payload));
        msg.destinationName = topic;
        STATE.mqtt.client.send(msg);
        return true;
    } catch { return false; }
}

function handleMQTTMessage(topic, payload) {
    for (const [id, topics] of Object.entries(STATE.deviceTopics)) {
        if (topics.sub === topic) {
            try {
                const data = JSON.parse(payload);
                const ns   = data.state === 1 || data.state === true || data.state === 'on';
                const prev = STATE.deviceStates[id];
                STATE.deviceStates[id] = ns;
                if (ns !== prev) {
                    if (ns) STATE.deviceOnAt[id] = Date.now();
                    else    delete STATE.deviceOnAt[id];
                    addLog(STATE.devices[id]?.name, `Status: ${ns ? 'ON' : 'OFF'}`, 'MQTT', 'info');
                }
            } catch {
                const raw              = payload.toLowerCase().trim();
                STATE.deviceStates[id] = raw === '1' || raw === 'on';
            }
            updateDeviceUI(id);
            updateDashboardStats();
            return;
        }
    }
    for (const [id, sensor] of Object.entries(STATE.sensors)) {
        if (sensor.topic === topic) {
            try {
                const data = JSON.parse(payload);
                const val  = typeof data.value !== 'undefined'
                    ? parseFloat(data.value)
                    : (typeof data === 'number' ? data : parseFloat(payload));
                if (!isNaN(val)) processSensorValue(id, val);
            } catch {
                const val = parseFloat(payload);
                if (!isNaN(val)) processSensorValue(id, val);
            }
            return;
        }
    }
}

function processSensorValue(sensorId, val) {
    STATE.sensorData[sensorId] = val;
    if (!STATE.sensorHistory[sensorId]) STATE.sensorHistory[sensorId] = [];
    STATE.sensorHistory[sensorId].push(val);
    if (STATE.sensorHistory[sensorId].length > 20) STATE.sensorHistory[sensorId].shift();

    const sensor = STATE.sensors[sensorId];
    if (sensor?.type === 'presence') updateSensorPresenceUI(sensorId);
    else                             updateSensorValueUI(sensorId);

    document.getElementById(`sensor-card-${sensorId}`)?.classList.add('has-data');
    checkAutomationRules();

    if (typeof automationEngine !== 'undefined' && automationEngine.isActive)
        automationEngine.evaluateSensorRules(sensorId, val);

    updateDashboardStats();
}

/* ==================== DEVICE UI ==================== */
function updateDeviceUI(deviceId) {
    const id   = String(deviceId);
    const isOn = STATE.deviceStates[id];
    const g    = s => document.getElementById(s);

    g(`card-${id}`)?.classList.toggle('on', isOn);
    g(`dot-${id}`)?.classList.toggle('on', isOn);
    g(`icon-${id}`)?.classList.toggle('on', isOn);

    const tog = g(`device-toggle-${id}`);
    if (tog) tog.checked = isOn;

    g(`row-${id}`)?.classList.toggle('on', isOn);
    const lbl = g(`lbl-${id}`);
    if (lbl) { lbl.textContent = isOn ? 'Aktif' : 'Mati'; lbl.className = 'status-text' + (isOn ? ' on' : ''); }

    const qc = g(`qc-${id}`);
    if (qc) {
        qc.classList.toggle('on', isOn);
        const qs = qc.querySelector('.qc-status'); if (qs) qs.textContent = isOn ? 'Aktif' : 'Mati';
        const qt = qc.querySelector('input[type=checkbox]'); if (qt) qt.checked = isOn;
    }

    // Update speed control if fan
    const speedRow = g(`speed-row-${id}`);
    if (speedRow) speedRow.style.display = isOn ? 'flex' : 'none';
}

function toggleDeviceState(deviceId, newState) {
    const id   = String(deviceId);
    const prev = STATE.deviceStates[id];
    STATE.deviceStates[id] = newState;
    if (newState && !prev) STATE.deviceOnAt[id] = Date.now();
    else if (!newState)    delete STATE.deviceOnAt[id];
    updateDeviceUI(id);
    const t = STATE.deviceTopics[id];
    if (t?.pub) publishMQTT(t.pub, { state: newState ? 1 : 0 });
    addLog(STATE.devices[id]?.name, newState ? 'Dinyalakan' : 'Dimatikan', 'Manual', 'info');
    updateDashboardStats();
}

function applyDeviceState(deviceId, newState, reason = 'Automation') {
    const id = String(deviceId);
    if (STATE.deviceStates[id] === newState) return;
    STATE.deviceStates[id] = newState;
    if (newState) STATE.deviceOnAt[id] = Date.now();
    else          delete STATE.deviceOnAt[id];
    updateDeviceUI(id);
    const t = STATE.deviceTopics[id];
    if (t?.pub) publishMQTT(t.pub, { state: newState ? 1 : 0 });
    addLog(STATE.devices[id]?.name, `${newState ? 'ON' : 'OFF'} (${reason})`, 'Automation', newState ? 'success' : 'info');
    updateDashboardStats();
}

// Fan speed control
function setFanSpeed(deviceId, speed) {
    const id = String(deviceId);
    const t  = STATE.deviceTopics[id];
    if (t?.pub) publishMQTT(t.pub, { state: 1, speed: parseInt(speed) });
    addLog(STATE.devices[id]?.name, `Kecepatan kipas: ${speed}%`, 'Manual', 'info');
    // Update slider display
    const sl = document.getElementById(`fan-speed-${id}`);
    const lb = document.getElementById(`fan-speed-label-${id}`);
    if (sl) sl.value = speed;
    if (lb) lb.textContent = speed + '%';
}

/* ==================== RENDER DEVICES ==================== */
function renderDevices() {
    const grid  = document.getElementById('devicesGrid');
    const empty = document.getElementById('emptyDevices');
    if (!grid) return;
    grid.innerHTML = '';
    const keys = Object.keys(STATE.devices);
    if (empty) empty.classList.toggle('hidden', keys.length > 0);
    keys.forEach(id => {
        const device = STATE.devices[id];
        const isOn   = STATE.deviceStates[id];
        const isFan  = device.icon === 'fa-wind' || device.name?.toLowerCase().includes('kipas') || device.name?.toLowerCase().includes('fan');
        const card   = document.createElement('div');
        card.id        = `card-${id}`;
        card.className = 'device-card' + (isOn ? ' on' : '');

        const fanControls = isFan ? `
            <div id="speed-row-${id}" class="fan-speed-row" style="display:${isOn ? 'flex' : 'none'}">
                <i class="fas fa-wind" style="font-size:11px;color:var(--ink-4)"></i>
                <input type="range" id="fan-speed-${id}" class="fan-slider"
                    min="0" max="100" step="25" value="50"
                    oninput="setFanSpeed('${id}', this.value); document.getElementById('fan-speed-label-${id}').textContent=this.value+'%'">
                <span id="fan-speed-label-${id}" class="fan-speed-val">50%</span>
            </div>` : '';

        card.innerHTML = `
            <div class="device-card-top">
                <div class="device-card-info">
                    <div class="device-big-icon${isOn ? ' on' : ''}" id="icon-${id}">
                        <i class="fas ${device.icon || 'fa-plug'}"></i>
                    </div>
                    <div>
                        <div class="device-name">${escHtml(device.name)}</div>
                        <div class="device-type">${getDeviceTypeName(device.icon)}</div>
                    </div>
                </div>
                <div class="device-card-actions">
                    <button class="icon-btn small" onclick="openTopicSettings('${id}')" title="Setting">
                        <i class="fas fa-sliders"></i>
                    </button>
                    <button class="trash-btn" onclick="removeDevice('${id}')" title="Hapus">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="device-status-row${isOn ? ' on' : ''}" id="row-${id}">
                <div class="device-status-left">
                    <span class="status-dot${isOn ? ' on' : ''}" id="dot-${id}"></span>
                    <div>
                        <span class="status-text${isOn ? ' on' : ''}" id="lbl-${id}">${isOn ? 'Aktif' : 'Mati'}</span>
                        <div class="device-duration${isOn ? ' on' : ''}" id="dur-${id}">${isOn ? 'Baru nyala' : 'Mati'}</div>
                    </div>
                </div>
                <label class="toggle-wrapper" onclick="event.stopPropagation()">
                    <input type="checkbox" id="device-toggle-${id}" class="toggle-input"
                        ${isOn ? 'checked' : ''}
                        onchange="toggleDeviceState('${id}', this.checked)">
                    <span class="toggle-track"></span>
                </label>
            </div>
            ${fanControls}`;
        grid.appendChild(card);
    });
}

function getDeviceTypeName(icon) {
    const map = {
        'fa-lightbulb':  'Lampu',
        'fa-wind':       'Kipas Angin',
        'fa-snowflake':  'AC / Pendingin',
        'fa-tv':         'Televisi',
        'fa-lock':       'Kunci Pintu',
        'fa-door-open':  'Pintu',
        'fa-video':      'Kamera CCTV',
        'fa-volume-up':  'Speaker',
        'fa-plug':       'Stop Kontak'
    };
    return map[icon] || 'Perangkat IoT';
}

function filterDevices(q) {
    const lq = q.toLowerCase();
    document.querySelectorAll('.device-card').forEach(c => {
        const id        = c.id.replace('card-', '');
        c.style.display = (STATE.devices[id]?.name?.toLowerCase() || '').includes(lq) ? '' : 'none';
    });
}

/* ==================== SENSORS ==================== */
const SENSOR_CONFIG = {
    temperature: { icon: 'fa-temperature-half', min: 20,  max: 35,  barClass: 'temp-bar' },
    humidity:    { icon: 'fa-droplet',           min: 0,   max: 100, barClass: 'humidity-bar' },
    air_quality: { icon: 'fa-wind',              min: 0,   max: 500, barClass: 'air-bar' },
    presence:    { icon: 'fa-user-check' },
    brightness:  { icon: 'fa-sun',               min: 0,   max: 1,   barClass: 'brightness-bar' },
    motion:      { icon: 'fa-person-running' },
    smoke:       { icon: 'fa-fire',              min: 0,   max: 100, barClass: 'default-bar' },
    gas:         { icon: 'fa-triangle-exclamation', min: 0, max: 100, barClass: 'default-bar' }
};

const SENSOR_LABELS = {
    temperature: 'Suhu',      humidity: 'Kelembaban',   air_quality: 'Kualitas Udara',
    presence:    'Kehadiran', brightness: 'Kecerahan',  motion: 'Gerakan',
    smoke:       'Asap',      gas: 'Gas'
};

function updateSensorValueUI(sensorId) {
    const val    = STATE.sensorData[sensorId];
    const sensor = STATE.sensors[sensorId];
    if (val === null || val === undefined || !sensor) return;
    const valEl  = document.getElementById(`val-${sensorId}`);
    const barEl  = document.getElementById(`bar-${sensorId}`);
    const timeEl = document.getElementById(`time-${sensorId}`);
    if (valEl)  valEl.textContent  = val.toFixed(sensor.type === 'brightness' ? 2 : 1) + (sensor.unit || '');
    if (timeEl) timeEl.textContent = new Date().toLocaleTimeString('id-ID');
    if (barEl) {
        const cfg = SENSOR_CONFIG[sensor.type] || {};
        if (cfg.min !== undefined && cfg.max !== undefined) {
            barEl.style.width = Math.min(100, Math.max(0,
                ((val - cfg.min) / (cfg.max - cfg.min)) * 100
            )) + '%';
        }
    }
    drawSparkline(sensorId);
}

function updateSensorPresenceUI(sensorId) {
    const val  = STATE.sensorData[sensorId];
    const card = document.getElementById(`sensor-card-${sensorId}`);
    const dot  = document.getElementById(`presence-dot-${sensorId}`);
    const txt  = document.getElementById(`presence-txt-${sensorId}`);
    const time = document.getElementById(`time-${sensorId}`);
    if (dot)  dot.classList.toggle('detected', !!val);
    if (txt)  txt.textContent  = val ? 'Terdeteksi' : 'Tidak Terdeteksi';
    if (time) time.textContent = new Date().toLocaleTimeString('id-ID');
    if (card) card.classList.toggle('has-data', !!val);
}

function drawSparkline(sensorId) {
    const canvas  = document.getElementById(`spark-${sensorId}`);
    if (!canvas) return;
    const history = STATE.sensorHistory[sensorId] || [];
    if (history.length < 2) return;
    const W = canvas.clientWidth  || canvas.offsetWidth  || 200;
    const H = canvas.clientHeight || canvas.offsetHeight || 36;
    canvas.width  = W;
    canvas.height = H;
    const ctx   = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    const mn    = Math.min(...history);
    const mx    = Math.max(...history);
    const range = mx - mn || 1;
    const pts   = history.map((v, i) => ({
        x: (i / (history.length - 1)) * W,
        y: H - ((v - mn) / range) * (H - 4) - 2
    }));
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, 'rgba(99,102,241,.18)');
    grad.addColorStop(1, 'rgba(99,102,241,0)');
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 1.5; ctx.lineJoin = 'round'; ctx.stroke();
    const last = pts[pts.length - 1];
    ctx.beginPath(); ctx.arc(last.x, last.y, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#6366f1'; ctx.fill();
}

function renderSensors() {
    const grid  = document.getElementById('sensorsGrid');
    const empty = document.getElementById('emptySensors');
    if (!grid) return;
    grid.innerHTML = '';
    const keys = Object.keys(STATE.sensors);
    if (empty) empty.classList.toggle('hidden', keys.length > 0);
    keys.forEach(sensorId => {
        const sensor    = STATE.sensors[sensorId];
        const val       = STATE.sensorData[sensorId];
        const cfg       = SENSOR_CONFIG[sensor.type] || {};
        const typeLabel = SENSOR_LABELS[sensor.type] || sensor.type;
        const card      = document.createElement('div');
        card.id         = `sensor-card-${sensorId}`;
        card.className  = 'sensor-card' + (val !== null && val !== undefined ? ' has-data' : '');

        if (sensor.type === 'presence') {
            const isD = !!val;
            card.innerHTML = `
                <div class="sensor-card-top">
                    <div class="sensor-card-info">
                        <div class="sensor-big-icon"><i class="fas ${sensor.icon || cfg.icon || 'fa-microchip'}"></i></div>
                        <div>
                            <div class="sensor-name">${escHtml(sensor.name)}</div>
                            <div class="sensor-type-label">${typeLabel}</div>
                        </div>
                    </div>
                    <div class="sensor-card-actions">
                        <button class="icon-btn small" onclick="openSensorSettings('${sensorId}')"><i class="fas fa-sliders"></i></button>
                        <button class="trash-btn" onclick="removeSensor('${sensorId}')"><i class="fas fa-trash"></i></button>
                    </div>
                </div>
                <div class="presence-status">
                    <div class="presence-dot${isD ? ' detected' : ''}" id="presence-dot-${sensorId}"></div>
                    <div class="presence-text" id="presence-txt-${sensorId}">${isD ? 'Terdeteksi' : 'Tidak Terdeteksi'}</div>
                </div>
                <div class="sensor-meta">
                    <span>Boolean</span>
                    <span class="sensor-meta-val" id="time-${sensorId}">—</span>
                </div>`;
        } else {
            let pct = 0;
            const hasVal = val !== null && val !== undefined;
            if (hasVal && cfg.min !== undefined && cfg.max !== undefined)
                pct = Math.min(100, Math.max(0, ((val - cfg.min) / (cfg.max - cfg.min)) * 100));
            const valStr = hasVal
                ? val.toFixed(sensor.type === 'brightness' ? 2 : 1) + (sensor.unit || '')
                : '—';
            card.innerHTML = `
                <div class="sensor-card-top">
                    <div class="sensor-card-info">
                        <div class="sensor-big-icon"><i class="fas ${sensor.icon || cfg.icon || 'fa-microchip'}"></i></div>
                        <div>
                            <div class="sensor-name">${escHtml(sensor.name)}</div>
                            <div class="sensor-type-label">${typeLabel}</div>
                        </div>
                    </div>
                    <div class="sensor-card-actions">
                        <button class="icon-btn small" onclick="openSensorSettings('${sensorId}')"><i class="fas fa-sliders"></i></button>
                        <button class="trash-btn" onclick="removeSensor('${sensorId}')"><i class="fas fa-trash"></i></button>
                    </div>
                </div>
                <div class="sensor-status-row">
                    <span class="sensor-value-big" id="val-${sensorId}">${valStr}</span>
                    <span class="sensor-meta-val" id="time-${sensorId}">—</span>
                </div>
                <canvas class="sparkline-canvas" id="spark-${sensorId}"></canvas>
                <div class="progress-track">
                    <div class="progress-fill ${cfg.barClass || 'default-bar'}" id="bar-${sensorId}" style="width:${pct}%"></div>
                </div>
                <div class="progress-labels">
                    <span>${cfg.min !== undefined ? cfg.min + (sensor.unit || '') : 'Min'}</span>
                    <span>${cfg.max !== undefined ? cfg.max + (sensor.unit || '') : 'Max'}</span>
                </div>`;
        }
        grid.appendChild(card);
        requestAnimationFrame(() => drawSparkline(sensorId));
    });
}

function filterSensors(q) {
    const lq = q.toLowerCase();
    document.querySelectorAll('.sensor-card').forEach(c => {
        const id        = c.id.replace('sensor-card-', '');
        c.style.display = (STATE.sensors[id]?.name?.toLowerCase() || '').includes(lq) ? '' : 'none';
    });
}

/* ==================== QUICK CONTROLS ==================== */
function renderQuickControls() {
    const container = document.getElementById('quickControlsContainer');
    if (!container) return;
    container.innerHTML = '';
    const devices = STATE.quickControlDevices
        .map(id => [String(id), STATE.devices[String(id)]])
        .filter(([, d]) => !!d);

    if (!devices.length) {
        container.innerHTML = `
            <div style="text-align:center;padding:36px 16px;color:var(--ink-4)">
                <i class="fas fa-hand-pointer" style="font-size:22px;margin-bottom:10px;display:block;opacity:.25"></i>
                <p style="font-size:12px;margin-bottom:10px">Belum ada perangkat dipilih</p>
                <button onclick="openQuickControlSettings()"
                    style="font-size:11px;color:var(--a);background:none;border:none;cursor:pointer;font-family:inherit">
                    Pilih perangkat →
                </button>
            </div>`;
        return;
    }
    devices.forEach(([id, device]) => {
        const isOn = STATE.deviceStates[id];
        const item = document.createElement('div');
        item.className = 'qc-item' + (isOn ? ' on' : '');
        item.id        = `qc-${id}`;
        item.onclick   = () => toggleDeviceState(id, !STATE.deviceStates[id]);
        item.innerHTML = `
            <div class="qc-info">
                <div class="qc-icon"><i class="fas ${device.icon || 'fa-plug'}"></i></div>
                <div>
                    <div class="qc-name">${escHtml(device.name)}</div>
                    <div class="qc-status">${isOn ? 'Aktif' : 'Mati'}</div>
                </div>
            </div>
            <label class="toggle-wrapper" onclick="event.stopPropagation()">
                <input type="checkbox" class="toggle-input" ${isOn ? 'checked' : ''}
                    onchange="toggleDeviceState('${id}', this.checked)">
                <span class="toggle-track"></span>
            </label>`;
        container.appendChild(item);
    });
}

function openQuickControlSettings() {
    const modal = document.getElementById('quickControlModal');
    const list  = document.getElementById('quickControlDevicesList');
    if (!modal || !list) return;
    list.innerHTML = '';
    const keys = Object.keys(STATE.devices);
    if (!keys.length) {
        list.innerHTML = '<p class="modal-loading">Belum ada perangkat. Tambahkan dulu di menu Perangkat.</p>';
        modal.classList.add('show');
        return;
    }
    keys.forEach(id => {
        const device  = STATE.devices[id];
        const checked = STATE.quickControlDevices.map(String).includes(String(id));
        const item    = document.createElement('label');
        item.className = 'modal-item' + (checked ? ' selected' : '');
        item.innerHTML = `
            <input type="checkbox" value="${id}" class="qc-picker-checkbox" ${checked ? 'checked' : ''}
                style="display:none"
                onchange="this.closest('label').classList.toggle('selected', this.checked)">
            <i class="fas ${device.icon || 'fa-plug'}"></i>
            <span style="font-size:13px;font-weight:600;flex:1">${escHtml(device.name)}</span>
            ${checked ? '<i class="fas fa-check" style="color:var(--a);font-size:11px"></i>' : ''}`;
        list.appendChild(item);
    });
    modal.classList.add('show');
}

function closeQuickControlSettings() {
    document.getElementById('quickControlModal')?.classList.remove('show');
}

async function saveQuickControlSettings() {
    const sel = Array.from(document.querySelectorAll('.qc-picker-checkbox:checked')).map(c => c.value);
    if (sel.length > 4) { showToast('Maksimal 4 perangkat!', 'warning'); return; }
    STATE.quickControlDevices = sel;
    await apiPost('save_settings', { quick_control_devices: sel });
    renderQuickControls();
    closeQuickControlSettings();
    showToast('Quick Control diperbarui!', 'success');
}

/* ==================== AUTOMATION — SENSOR BASED ==================== */
const SENSOR_AUTO_META = {
    temperature: {
        icon: 'fa-temperature-half', color: 'var(--red)',    bg: 'var(--red-bg)',    label: 'Suhu',          unit: '°C',
        conditions: [{ key: 'gt', label: 'Lebih dari (>)', defaultVal: 28 }, { key: 'lt', label: 'Kurang dari (<)', defaultVal: 24 }, { key: 'range', label: 'Di luar rentang' }]
    },
    humidity: {
        icon: 'fa-droplet',          color: 'var(--blue)',   bg: 'var(--blue-bg)',   label: 'Kelembaban',    unit: '%',
        conditions: [{ key: 'gt', label: 'Lebih dari (>)', defaultVal: 75 }, { key: 'lt', label: 'Kurang dari (<)', defaultVal: 35 }]
    },
    brightness: {
        icon: 'fa-sun',              color: 'var(--amber)',  bg: 'var(--amber-bg)',  label: 'Kecerahan',     unit: 'lux',
        conditions: [{ key: 'lt', label: 'Lebih gelap dari (<)', defaultVal: 0.35 }, { key: 'gt', label: 'Lebih terang dari (>)', defaultVal: 0.6 }]
    },
    presence: {
        icon: 'fa-user-check',       color: 'var(--green)',  bg: 'var(--green-bg)',  label: 'Kehadiran',     unit: '',
        conditions: [{ key: 'detected', label: 'Terdeteksi' }, { key: 'absent', label: 'Tidak Ada' }]
    },
    motion: {
        icon: 'fa-person-running',   color: 'var(--purple)', bg: 'var(--purple-bg)', label: 'Gerakan',       unit: '',
        conditions: [{ key: 'detected', label: 'Ada Gerakan' }, { key: 'absent', label: 'Tidak Ada Gerakan' }]
    },
    air_quality: {
        icon: 'fa-wind',             color: '#0d9488',       bg: '#f0fdfa',          label: 'Kualitas Udara', unit: 'AQI',
        conditions: [{ key: 'gt', label: 'Lebih dari (>)', defaultVal: 150 }, { key: 'lt', label: 'Kurang dari (<)', defaultVal: 50 }]
    },
    smoke: {
        icon: 'fa-fire',             color: 'var(--red)',    bg: 'var(--red-bg)',    label: 'Asap',          unit: 'ppm',
        conditions: [{ key: 'gt', label: 'Terdeteksi (>)', defaultVal: 50 }]
    },
    gas: {
        icon: 'fa-triangle-exclamation', color: 'var(--amber)', bg: 'var(--amber-bg)', label: 'Gas',         unit: 'ppm',
        conditions: [{ key: 'gt', label: 'Terdeteksi (>)', defaultVal: 200 }]
    }
};

function initAutomationRules() {
    if (!STATE.automationRules) STATE.automationRules = {};
    try {
        const saved = localStorage.getItem('iotzy_auto_rules');
        if (saved) STATE.automationRules = JSON.parse(saved);
    } catch (_) {}
}

function saveAutomationRules() {
    try { localStorage.setItem('iotzy_auto_rules', JSON.stringify(STATE.automationRules)); } catch (_) {}
}

function getRulesForSensor(sensorId) {
    return STATE.automationRules[String(sensorId)] || [];
}

function addAutomationRule(sensorId, rule) {
    const id = String(sensorId);
    if (!STATE.automationRules[id]) STATE.automationRules[id] = [];
    rule.ruleId    = `r_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`;
    rule.enabled   = true;
    rule.lastFired = null;
    STATE.automationRules[id].push(rule);
    saveAutomationRules();
    renderAutomationView();
    showToast('Aturan ditambahkan!', 'success');
}

function removeAutomationRule(sensorId, ruleId) {
    const id = String(sensorId);
    if (!STATE.automationRules[id]) return;
    STATE.automationRules[id] = STATE.automationRules[id].filter(r => r.ruleId !== ruleId);
    if (!STATE.automationRules[id].length) delete STATE.automationRules[id];
    saveAutomationRules();
    renderAutomationView();
    showToast('Aturan dihapus', 'info');
}

function toggleAutomationRule(sensorId, ruleId, enabled) {
    const rule = (STATE.automationRules[String(sensorId)] || []).find(r => r.ruleId === ruleId);
    if (rule) { rule.enabled = enabled; saveAutomationRules(); }
}

function renderAutomationView() {
    initAutomationRules();
    const grid  = document.getElementById('automationGrid');
    const empty = document.getElementById('emptyAutomation');
    if (!grid) return;
    grid.innerHTML = '';
    const sensorKeys = Object.keys(STATE.sensors);
    if (!sensorKeys.length) { if (empty) empty.classList.remove('hidden'); return; }
    if (empty) empty.classList.add('hidden');

    sensorKeys.forEach(sensorId => {
        const sensor  = STATE.sensors[sensorId];
        const meta    = SENSOR_AUTO_META[sensor.type] || SENSOR_AUTO_META.temperature;
        const rules   = getRulesForSensor(sensorId);
        const curVal  = STATE.sensorData[sensorId];
        const valBadge = (curVal !== null && curVal !== undefined)
            ? `<span class="auto-val-badge">${parseFloat(curVal).toFixed(1)}${sensor.unit || meta.unit}</span>`
            : `<span style="font-size:11px;color:var(--ink-4)">Menunggu data…</span>`;

        const card       = document.createElement('div');
        card.className   = 'auto-card';
        card.id          = `autocard-${sensorId}`;
        card.innerHTML   = `
            <div class="auto-card-head">
                <div class="auto-card-head-l">
                    <div class="auto-icon" style="background:${meta.bg};color:${meta.color}">
                        <i class="fas ${meta.icon}"></i>
                    </div>
                    <div>
                        <div class="auto-card-title">${escHtml(sensor.name)}</div>
                        <div class="auto-card-sub">${meta.label} · ${valBadge}</div>
                    </div>
                </div>
            </div>
            <div class="auto-card-body">
                <div style="font-size:10px;font-weight:700;color:var(--ink-4);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">
                    Aturan Aktif
                </div>
                <div id="rules-list-${sensorId}" style="display:flex;flex-direction:column;gap:5px">
                    ${rules.length === 0
                        ? '<div style="font-size:11px;color:var(--ink-4);padding:8px;text-align:center;background:var(--surface-2);border-radius:6px">Belum ada aturan. Klik tambah di bawah.</div>'
                        : rules.map(r => buildRuleRowHTML(sensorId, r, sensor, meta)).join('')}
                </div>
                <button onclick="openAddRuleModal('${sensorId}')" class="btn-ghost full" style="margin-top:8px;font-size:12px">
                    <i class="fas fa-plus"></i> Tambah Aturan
                </button>
            </div>`;
        grid.appendChild(card);
    });
}

function buildRuleRowHTML(sensorId, rule, sensor, meta) {
    const device      = STATE.devices[String(rule.deviceId)];
    const devName     = device ? escHtml(device.name) : `<span style="color:var(--red)">Perangkat dihapus</span>`;
    const condLabel   = getConditionLabel(rule, sensor, meta);
    const actionBadge = rule.action === 'on'
        ? `<span class="rule-badge-on">ON</span>`
        : `<span class="rule-badge-off">OFF</span>`;
    const devIcon = device ? `<i class="fas ${device.icon || 'fa-plug'}" style="font-size:11px;color:var(--ink-3)"></i>` : '';
    return `
        <div class="rule-row" id="rule-row-${rule.ruleId}">
            <label class="toggle-wrapper" style="flex-shrink:0">
                <input type="checkbox" class="toggle-input" ${rule.enabled ? 'checked' : ''}
                    onchange="toggleAutomationRule('${sensorId}', '${rule.ruleId}', this.checked)">
                <span class="toggle-track"></span>
            </label>
            <div style="flex:1;min-width:0;font-size:11px">
                <div style="font-weight:600;color:var(--ink)">
                    ${condLabel} → ${actionBadge} ${devIcon} ${devName}
                </div>
                ${rule.delay > 0 ? `<div style="font-size:10px;color:var(--ink-4)">Delay: ${rule.delay}ms</div>` : ''}
            </div>
            <button onclick="removeAutomationRule('${sensorId}', '${rule.ruleId}')"
                class="trash-btn" title="Hapus"><i class="fas fa-trash"></i></button>
        </div>`;
}

function getConditionLabel(rule, sensor, meta) {
    const u = sensor.unit || meta.unit || '';
    switch (rule.condition) {
        case 'gt':       return `> ${rule.threshold}${u}`;
        case 'lt':       return `< ${rule.threshold}${u}`;
        case 'range':    return `di luar ${rule.thresholdMin}–${rule.thresholdMax}${u}`;
        case 'detected': return 'terdeteksi';
        case 'absent':   return 'tidak ada';
        default:         return rule.condition;
    }
}

let _addRuleSensorId = null;

function openAddRuleModal(sensorId) {
    _addRuleSensorId = String(sensorId);
    const sensor = STATE.sensors[_addRuleSensorId];
    const meta   = SENSOR_AUTO_META[sensor.type] || SENSOR_AUTO_META.temperature;
    const modal  = document.getElementById('addRuleModal');
    if (!modal) return;

    document.getElementById('addRuleSensorLabel').textContent = sensor.name;
    const iconEl = document.getElementById('addRuleSensorIcon');
    if (iconEl) {
        iconEl.innerHTML   = `<i class="fas ${meta.icon}"></i>`;
        iconEl.style.cssText = `background:${meta.bg};color:${meta.color};width:34px;height:34px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0`;
    }

    const condSel     = document.getElementById('addRuleCondition');
    condSel.innerHTML = meta.conditions.map(c => `<option value="${c.key}">${c.label}</option>`).join('');

    const devSel  = document.getElementById('addRuleDevice');
    const devKeys = Object.keys(STATE.devices);
    devSel.innerHTML = devKeys.length
        ? devKeys.map(id => `<option value="${id}">${escHtml(STATE.devices[id].name)}</option>`).join('')
        : '<option disabled>Belum ada perangkat</option>';

    updateRuleConditionUI(condSel.value, meta);
    condSel.onchange = () => updateRuleConditionUI(condSel.value, meta);
    document.getElementById('addRuleDelay').value = 0;
    modal.classList.add('show');
}

function updateRuleConditionUI(cond, meta) {
    const tRow = document.getElementById('addRuleThresholdRow');
    const rRow = document.getElementById('addRuleRangeRow');
    if (!tRow || !rRow) return;
    if (cond === 'range') {
        tRow.style.display = 'none'; rRow.style.display = '';
    } else if (cond === 'detected' || cond === 'absent') {
        tRow.style.display = 'none'; rRow.style.display = 'none';
    } else {
        tRow.style.display = ''; rRow.style.display = 'none';
        const c = meta.conditions.find(x => x.key === cond);
        if (c && c.defaultVal != null) document.getElementById('addRuleThreshold').value = c.defaultVal;
        const sensor = _addRuleSensorId ? STATE.sensors[_addRuleSensorId] : null;
        document.getElementById('addRuleUnit').textContent = sensor?.unit || meta.unit || '';
    }
}

function closeAddRuleModal() {
    document.getElementById('addRuleModal')?.classList.remove('show');
    _addRuleSensorId = null;
}

function saveNewAutomationRule() {
    if (!_addRuleSensorId) return;
    const cond     = document.getElementById('addRuleCondition').value;
    const deviceId = document.getElementById('addRuleDevice').value;
    const action   = document.getElementById('addRuleAction').value;
    const delay    = parseInt(document.getElementById('addRuleDelay').value) || 0;

    if (!deviceId || !STATE.devices[String(deviceId)]) {
        showToast('Pilih perangkat terlebih dahulu!', 'warning'); return;
    }
    const rule = { condition: cond, deviceId: String(deviceId), action, delay };

    if (cond === 'range') {
        rule.thresholdMin = parseFloat(document.getElementById('addRuleThresholdMin').value);
        rule.thresholdMax = parseFloat(document.getElementById('addRuleThresholdMax').value);
        if (isNaN(rule.thresholdMin) || isNaN(rule.thresholdMax)) {
            showToast('Isi rentang nilai!', 'warning'); return;
        }
    } else if (cond !== 'detected' && cond !== 'absent') {
        rule.threshold = parseFloat(document.getElementById('addRuleThreshold').value);
        if (isNaN(rule.threshold)) { showToast('Isi nilai threshold!', 'warning'); return; }
    }
    addAutomationRule(_addRuleSensorId, rule);
    closeAddRuleModal();
}

function checkAutomationRules() {
    Object.keys(STATE.sensors).forEach(sensorId => {
        const val = STATE.sensorData[sensorId];
        if (val === null || val === undefined) return;
        getRulesForSensor(sensorId).forEach(rule => {
            if (!rule.enabled) return;
            if (shouldFireRule(rule, val)) fireRule(sensorId, rule);
        });
    });
}

function shouldFireRule(rule, val) {
    const v = parseFloat(val);
    switch (rule.condition) {
        case 'gt':       return v >  parseFloat(rule.threshold);
        case 'lt':       return v <  parseFloat(rule.threshold);
        case 'range':    return v <  parseFloat(rule.thresholdMin) || v > parseFloat(rule.thresholdMax);
        case 'detected': return !!val;
        case 'absent':   return !val;
        default:         return false;
    }
}

function fireRule(sensorId, rule) {
    const now      = Date.now();
    const cooldown = (rule.delay || 0) + 3000;
    if (rule.lastFired && now - rule.lastFired < cooldown) return;
    const deviceId = String(rule.deviceId);
    if (!STATE.devices[deviceId]) return;
    const newState = rule.action === 'on';
    if (STATE.deviceStates[deviceId] === newState) return;
    rule.lastFired = now;
    const sensorName = STATE.sensors[sensorId]?.name || 'Sensor';
    if (rule.delay > 0) {
        setTimeout(() => {
            if (STATE.deviceStates[deviceId] !== newState)
                applyDeviceState(deviceId, newState, `Auto(${sensorName})`);
        }, rule.delay);
    } else {
        applyDeviceState(deviceId, newState, `Auto(${sensorName})`);
    }
}

/* ==================== CV PERSON COUNT AUTOMATION ==================== */
// Called by cv-detector when person count changes
function onCVPersonCountUpdate(count) {
    STATE.cv.personCount   = count;
    STATE.cv.personPresent = count > 0;

    // Update all person count displays
    const g = id => document.getElementById(id);
    if (g('cvPersonCountBig'))   g('cvPersonCountBig').textContent   = count;
    if (g('statPersonCount'))    g('statPersonCount').textContent    = count;
    if (g('cvHumanCount'))       g('cvHumanCount').textContent       = count;
    if (g('statPersonPresence')) g('statPersonPresence').textContent = count > 0 ? `${count} orang` : 'Tidak ada';

    updateDashboardStats();
}

/* ==================== LOGGING ==================== */
async function loadLogs() {
    const result = await apiPost('get_logs', {});
    if (result && Array.isArray(result)) {
        STATE.logs = result.map(r => ({
            tanggal:  r.tanggal  || new Date(r.created_at).toLocaleDateString('id-ID'),
            waktu:    r.waktu    || new Date(r.created_at).toLocaleTimeString('id-ID'),
            device:   r.device_name,
            activity: r.activity,
            trigger:  r.trigger_type,
            type:     r.log_type,
            ts:       new Date(r.created_at).getTime()
        }));
        updateLogDisplay();
    }
}

async function addLog(device, activity, trigger, type = 'info') {
    const now = new Date();
    const log = {
        tanggal:  now.toLocaleDateString('id-ID'),
        waktu:    now.toLocaleTimeString('id-ID'),
        device:   device || 'System',
        activity, trigger, type,
        ts: now.getTime()
    };
    STATE.logs.unshift(log);
    if (STATE.logs.length > CONFIG.app.maxLogs) STATE.logs.length = CONFIG.app.maxLogs;
    updateLogDisplay();
    apiPost('add_log', { device: device || 'System', activity, trigger, type }).catch(() => {});
}

function updateLogDisplay() {
    const tbody = document.getElementById('logBody');
    const empty = document.getElementById('emptyLog');
    if (!tbody) return;
    const q        = STATE.logFilter.toLowerCase();
    const tf       = STATE.logTypeFilter;
    let filtered   = STATE.logs;
    if (q)  filtered = filtered.filter(l =>
        (l.device   || '').toLowerCase().includes(q) ||
        (l.activity || '').toLowerCase().includes(q) ||
        (l.trigger  || '').toLowerCase().includes(q));
    if (tf !== 'all') filtered = filtered.filter(l => l.type === tf);
    if (empty) empty.classList.toggle('hidden', filtered.length > 0);
    tbody.innerHTML = '';
    filtered.forEach(log => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td class="log-time">${escHtml(log.tanggal)}</td>
            <td class="log-time">${escHtml(log.waktu)}</td>
            <td style="font-size:12px;font-weight:600">${escHtml(log.device || '')}</td>
            <td style="font-size:12px">${escHtml(log.activity)}</td>
            <td><span class="log-badge ${log.type || 'info'}">${escHtml(log.trigger)}</span></td>`;
        tbody.appendChild(row);
    });
}

function filterLogs(q)    { STATE.logFilter     = q;  updateLogDisplay(); }
function filterLogType(t) { STATE.logTypeFilter = t;  updateLogDisplay(); }

async function clearLogs() {
    if (!confirm('Hapus semua log aktivitas?')) return;
    const result = await apiPost('clear_logs', {});
    if (result?.success) { STATE.logs = []; updateLogDisplay(); showToast('Log dihapus', 'success'); }
}

function exportLogsToExcel() {
    if (!STATE.logs.length) { showToast('Tidak ada log', 'warning'); return; }
    const rows = [['Tanggal', 'Waktu', 'Perangkat', 'Aktivitas', 'Trigger', 'Tipe']];
    STATE.logs.forEach(l => rows.push([l.tanggal, l.waktu, l.device, l.activity, l.trigger, l.type]));
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(rows), 'Log');
    XLSX.writeFile(wb, `iotzy-log-${Date.now()}.xlsx`);
    showToast('Log diekspor!', 'success');
}

/* ==================== DEVICE CRUD ==================== */
function openTopicSettings(deviceId) {
    const id     = String(deviceId);
    const device = STATE.devices[id];
    const topics = STATE.deviceTopics[id] || {};
    if (!device) return;
    const g = i => document.getElementById(i);
    if (g('topicDeviceName')) g('topicDeviceName').textContent = device.name;
    if (g('editDeviceName'))  g('editDeviceName').value        = device.name  || '';
    if (g('editDeviceIcon'))  g('editDeviceIcon').value        = device.icon  || 'fa-plug';
    if (g('deviceTopicSub'))  g('deviceTopicSub').value        = topics.sub   || device.topic_sub || '';
    if (g('deviceTopicPub'))  g('deviceTopicPub').value        = topics.pub   || device.topic_pub || '';
    const modal = document.getElementById('topicModal');
    if (modal) { modal.dataset.deviceId = id; modal.classList.add('show'); }
}

function closeTopicSettings() {
    document.getElementById('topicModal')?.classList.remove('show');
}

async function saveDeviceSettings() {
    const modal = document.getElementById('topicModal');
    if (!modal) return;
    const id   = String(modal.dataset.deviceId);
    const name = document.getElementById('editDeviceName')?.value.trim();
    const icon = document.getElementById('editDeviceIcon')?.value;
    const sub  = document.getElementById('deviceTopicSub')?.value.trim();
    const pub  = document.getElementById('deviceTopicPub')?.value.trim();
    if (!name) { showToast('Nama perangkat harus diisi!', 'warning'); return; }

    const result = await apiPost('update_device', { id, name, icon, topic_sub: sub, topic_pub: pub });
    if (result?.success) {
        STATE.devices[id]      = { ...STATE.devices[id], name, icon, topic_sub: sub, topic_pub: pub };
        STATE.deviceTopics[id] = { sub, pub };
        if (STATE.mqtt.connected && sub) { try { STATE.mqtt.client.subscribe(sub); } catch (_) {} }
        renderDevices(); renderQuickControls(); closeTopicSettings();
        addLog(name, 'Setting diperbarui', 'User', 'info');
        showToast('Setting disimpan!', 'success');
    } else showToast('Gagal menyimpan setting', 'error');
}

function openAddDeviceModal() {
    ['newDeviceName', 'newDeviceTopicSub', 'newDeviceTopicPub'].forEach(id => {
        const el = document.getElementById(id); if (el) el.value = '';
    });
    document.getElementById('addDeviceModal')?.classList.add('show');
}

function closeAddDeviceModal() {
    document.getElementById('addDeviceModal')?.classList.remove('show');
}

async function saveNewDevice() {
    const name = document.getElementById('newDeviceName')?.value.trim();
    if (!name) { showToast('Nama perangkat harus diisi!', 'warning'); return; }
    const icon = document.getElementById('newDeviceIcon')?.value   || 'fa-plug';
    const sub  = document.getElementById('newDeviceTopicSub')?.value.trim();
    const pub  = document.getElementById('newDeviceTopicPub')?.value.trim();
    const result = await apiPost('add_device', { name, icon, topic_sub: sub, topic_pub: pub });
    if (result?.success) {
        const id               = String(result.id);
        STATE.devices[id]      = { id, name, icon, type: 'switch', device_key: result.device_key, topic_sub: sub, topic_pub: pub };
        STATE.deviceTopics[id] = { sub: sub || '', pub: pub || '' };
        STATE.deviceStates[id] = false;
        if (STATE.mqtt.connected && sub) { try { STATE.mqtt.client.subscribe(sub); } catch (_) {} }
        renderDevices(); renderQuickControls(); closeAddDeviceModal();
        addLog('System', `Perangkat "${name}" ditambahkan`, 'User', 'success');
        showToast(`"${name}" ditambahkan!`, 'success');
        updateDashboardStats();
    } else showToast('Gagal menambahkan perangkat', 'error');
}

async function removeDevice(deviceId) {
    const id     = String(deviceId);
    const device = STATE.devices[id];
    if (!confirm(`Hapus perangkat "${device?.name}"?`)) return;
    const result = await apiPost('delete_device', { id });
    if (result?.success !== false && result !== null) {
        delete STATE.devices[id];
        delete STATE.deviceStates[id];
        delete STATE.deviceTopics[id];
        delete STATE.deviceOnAt[id];
        STATE.quickControlDevices = STATE.quickControlDevices.filter(qid => String(qid) !== id);
        renderDevices(); renderQuickControls();
        showToast(`"${device?.name}" dihapus`, 'info');
        updateDashboardStats();
    }
}

/* ==================== SENSOR CRUD ==================== */
function openAddSensorModal() {
    ['newSensorName', 'newSensorTopic', 'newSensorUnit'].forEach(id => {
        const el = document.getElementById(id); if (el) el.value = '';
    });
    document.getElementById('addSensorModal')?.classList.add('show');
}

function closeAddSensorModal() {
    document.getElementById('addSensorModal')?.classList.remove('show');
}

async function saveNewSensor() {
    const name  = document.getElementById('newSensorName')?.value.trim();
    const topic = document.getElementById('newSensorTopic')?.value.trim();
    if (!name)  { showToast('Nama sensor harus diisi!', 'warning'); return; }
    if (!topic) { showToast('MQTT Topic harus diisi!',  'warning'); return; }
    const type   = document.getElementById('newSensorType')?.value || 'temperature';
    const unit   = document.getElementById('newSensorUnit')?.value.trim();
    const result = await apiPost('add_sensor', { name, type, topic, unit });
    if (result?.success) {
        const cfg = SENSOR_CONFIG[type] || {};
        const id  = String(result.id);
        STATE.sensors[id]       = { id, name, type, icon: cfg.icon || 'fa-microchip', unit, topic, sensor_key: result.sensor_key };
        STATE.sensorData[id]    = null;
        STATE.sensorHistory[id] = [];
        if (STATE.mqtt.connected && topic) { try { STATE.mqtt.client.subscribe(topic); } catch (_) {} }
        renderSensors(); closeAddSensorModal();
        showToast(`Sensor "${name}" ditambahkan!`, 'success');
        updateDashboardStats();
    } else showToast('Gagal menambahkan sensor', 'error');
}

function openSensorSettings(sensorId) {
    const id     = String(sensorId);
    const sensor = STATE.sensors[id];
    if (!sensor) return;
    const g = i => document.getElementById(i);
    if (g('ssSensorName')) g('ssSensorName').textContent = sensor.name;
    if (g('ssEditName'))   g('ssEditName').value  = sensor.name  || '';
    if (g('ssEditTopic'))  g('ssEditTopic').value = sensor.topic || '';
    if (g('ssEditUnit'))   g('ssEditUnit').value  = sensor.unit  || '';
    if (g('ssEditType'))   g('ssEditType').value  = sensor.type  || 'temperature';
    const modal = document.getElementById('sensorSettingModal');
    if (modal) { modal.dataset.sensorId = id; modal.classList.add('show'); }
}

function closeSensorSettings() {
    document.getElementById('sensorSettingModal')?.classList.remove('show');
}

async function saveSensorSettings() {
    const modal = document.getElementById('sensorSettingModal');
    if (!modal) return;
    const id    = String(modal.dataset.sensorId);
    const name  = document.getElementById('ssEditName')?.value.trim();
    const topic = document.getElementById('ssEditTopic')?.value.trim();
    const unit  = document.getElementById('ssEditUnit')?.value.trim();
    const type  = document.getElementById('ssEditType')?.value;
    if (!name)  { showToast('Nama sensor harus diisi!', 'warning'); return; }
    if (!topic) { showToast('MQTT Topic harus diisi!',  'warning'); return; }
    const result = await apiPost('update_sensor', { id, name, topic, unit, type });
    if (result?.success) {
        const cfg      = SENSOR_CONFIG[type] || {};
        const oldTopic = STATE.sensors[id]?.topic;
        STATE.sensors[id] = { ...STATE.sensors[id], name, topic, unit, type, icon: cfg.icon || STATE.sensors[id].icon };
        if (STATE.mqtt.connected && topic !== oldTopic) {
            try { STATE.mqtt.client.subscribe(topic); } catch (_) {}
        }
        renderSensors(); closeSensorSettings();
        showToast('Setting sensor disimpan!', 'success');
    } else showToast('Gagal menyimpan setting sensor', 'error');
}

async function removeSensor(sensorId) {
    const id = String(sensorId);
    const s  = STATE.sensors[id];
    if (!confirm(`Hapus sensor "${s?.name}"?`)) return;
    const result = await apiPost('delete_sensor', { id });
    if (result?.success !== false && result !== null) {
        delete STATE.sensors[id];
        delete STATE.sensorData[id];
        delete STATE.sensorHistory[id];
        delete STATE.automationRules[id];
        saveAutomationRules();
        renderSensors();
        showToast(`"${s?.name}" dihapus`, 'info');
        updateDashboardStats();
    }
}

/* ==================== SETTINGS ==================== */
async function saveProfile() {
    const fullName = document.getElementById('settingFullName')?.value.trim();
    const email    = document.getElementById('settingEmail')?.value.trim();
    if (!fullName) { showToast('Nama lengkap harus diisi!', 'warning'); return; }
    if (!email)    { showToast('Email harus diisi!', 'warning'); return; }
    const result = await apiPost('update_profile', { full_name: fullName, email });
    if (result?.success) showToast('Profil disimpan!', 'success');
    else showToast(result?.error || 'Gagal menyimpan profil', 'error');
}

async function changePassword() {
    const cur  = document.getElementById('oldPassword')?.value;
    const nw   = document.getElementById('newPassword')?.value;
    const conf = document.getElementById('confirmPassword')?.value;
    if (!cur)                { showToast('Password saat ini harus diisi!', 'warning'); return; }
    if (!nw || nw.length < 8){ showToast('Password baru minimal 8 karakter!', 'warning'); return; }
    if (nw !== conf)         { showToast('Konfirmasi password tidak cocok!', 'warning'); return; }
    const result = await apiPost('change_password', { current_password: cur, new_password: nw });
    if (result?.success) {
        showToast('Password berhasil diubah!', 'success');
        ['oldPassword', 'newPassword', 'confirmPassword'].forEach(id => {
            const el = document.getElementById(id); if (el) el.value = '';
        });
    } else showToast(result?.error || 'Gagal mengganti password', 'error');
}

function openMQTTConfigModal()  { document.getElementById('mqttConfigModal')?.classList.add('show'); }
function closeMQTTConfigModal() { document.getElementById('mqttConfigModal')?.classList.remove('show'); }

async function saveMQTTConfig() {
    const broker   = document.getElementById('mqttBroker')?.value.trim();
    const port     = parseInt(document.getElementById('mqttPort')?.value);
    const clientId = document.getElementById('mqttClientId')?.value.trim();
    const path     = document.getElementById('mqttPath')?.value.trim();
    const useSSL   = document.getElementById('mqttUseSSL')?.checked ? 1 : 0;
    const username = document.getElementById('mqttUsername')?.value.trim();
    if (!broker) { showToast('Broker URL harus diisi!', 'warning'); return; }
    const result = await apiPost('save_settings', {
        mqtt_broker:    broker,
        mqtt_port:      port,
        mqtt_client_id: clientId,
        mqtt_path:      path,
        mqtt_use_ssl:   useSSL,
        mqtt_username:  username
    });
    if (result?.success) {
        closeMQTTConfigModal();
        showToast('Konfigurasi disimpan, menghubungkan…', 'success');
        setTimeout(connectMQTT, 300);
    } else showToast('Gagal menyimpan konfigurasi', 'error');
}

/* ==================== CV — COMPUTER VISION ==================== */
async function initializeCV() {
    if (CV.modelLoading) { showToast('Model sedang dimuat…', 'info'); return; }
    if (CV.modelLoaded)  { showToast('Model sudah siap! Klik Mulai Deteksi.', 'info'); return; }

    CV.modelLoading = true;
    const btn       = document.getElementById('btnLoadModel');
    const badge     = document.getElementById('cvModelBadge');
    const loadBadge = document.getElementById('cvLoadingStatus');

    if (btn)       { btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Memuat…'; }
    if (badge)     { badge.textContent = 'Loading'; badge.className = 'cv-status-badge loading'; }
    if (loadBadge) loadBadge.classList.remove('hidden');
    setCVStatus('loading', 'Memuat model…');
    addLog('CV', 'Memuat model COCO-SSD lite_mobilenet_v2…', 'System', 'info');

    cvDetector.setCallbacks({
        _tag:    'app_error',
        onError: (msg) => {
            CV.modelLoading = false;
            CV.modelLoaded  = false;
            setCVStatus('error', 'Error');
            if (badge)     { badge.textContent = 'Error'; badge.className = 'cv-status-badge error'; }
            if (btn)       { btn.disabled = false; btn.innerHTML = '<i class="fas fa-brain"></i> Coba Lagi'; }
            if (loadBadge) loadBadge.classList.add('hidden');
            showToast('Gagal: ' + msg, 'error');
            addLog('CV', 'Error: ' + msg, 'System', 'error');
        }
    });

    const ok = await cvDetector.initialize();
    CV.modelLoading = false;

    if (ok) {
        CV.modelLoaded = true;
        CV.model       = cvDetector.model;
        setCVStatus('ready', 'Siap');
        if (badge)     { badge.textContent = 'Ready'; badge.className = 'cv-status-badge ready'; }
        if (loadBadge) loadBadge.classList.add('hidden');
        if (btn) {
            btn.disabled      = false;
            btn.innerHTML     = '<i class="fas fa-check-circle"></i> Model Siap';
            btn.style.cssText = 'background:var(--green-bg);color:var(--green);border-color:rgba(22,163,74,.2)';
        }
        document.getElementById('btnStartCV')?.removeAttribute('disabled');
        showToast('✅ Model COCO-SSD berhasil dimuat!', 'success');
        addLog('CV', 'Model siap (' + tf.getBackend() + ')', 'System', 'success');
        initCVOverlay();
        automationEngine.initialize();
        cvUI.initialize();
        cvUI.renderAutomationSettings();
    } else {
        showToast('Gagal memuat model. Periksa koneksi internet dan coba lagi.', 'error');
    }
}

function initCVOverlay() {
    cvUI.attachOverlay('cameraFocusContainer');
    let c = document.getElementById('cvOverlayCanvas');
    if (!c) {
        c          = document.createElement('canvas');
        c.id       = 'cvOverlayCanvas';
        c.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10';
        const cont = document.getElementById('cameraFocusContainer');
        if (cont) cont.appendChild(c);
    }
    CV.overlayCanvas = c;
    CV.overlayCtx    = c.getContext('2d');
}

function startCVDetection() {
    if (!CV.modelLoaded)           { showToast('Load model terlebih dahulu!', 'warning');    return; }
    if (!STATE.camera.stream)      { showToast('Aktifkan kamera terlebih dahulu!', 'warning'); return; }
    if (CV.detecting)              return;

    CV.detecting = true;
    const video    = document.getElementById('cameraFocus');
    const startBtn = document.getElementById('btnStartCV');
    const stopBtn  = document.getElementById('btnStopCV');
    const badge    = document.getElementById('cvModelBadge');

    if (startBtn) startBtn.disabled = true;
    if (stopBtn)  stopBtn.disabled  = false;
    if (badge)    { badge.textContent = 'Active'; badge.className = 'cv-status-badge active'; }
    setCVStatus('ready', 'Mendeteksi…');

    const hudEl = document.getElementById('cvDetectionInfo');
    if (hudEl) hudEl.style.display = 'flex';

    initCVOverlay();

    CV.frameCount = 0; CV.fps = 0;
    CV.fpsTimer   = setInterval(() => {
        CV.fps        = CV.frameCount;
        CV.frameCount = 0;
        const el = document.getElementById('cvFpsStatus');
        if (el) el.textContent = CV.fps + ' fps';
    }, 1000);

    if (video) {
        cvDetector.startDetection(video);
        lightAnalyzer.startAnalysis(video);
    }

    automationEngine.start();
    addLog('CV', 'Deteksi CV dimulai', 'System', 'success');
    showToast('▶️ Deteksi CV aktif!', 'success');
}

function stopCVDetection() {
    CV.detecting = false;
    cvDetector.stopDetection();
    lightAnalyzer.stopAnalysis();
    automationEngine.stop();

    if (CV.fpsTimer) { clearInterval(CV.fpsTimer); CV.fpsTimer = null; }
    cvUI.clearOverlay();

    // Reset person count
    STATE.cv.personCount  = 0;
    STATE.cv.personPresent = false;
    onCVPersonCountUpdate(0);

    const startBtn = document.getElementById('btnStartCV');
    const stopBtn  = document.getElementById('btnStopCV');
    const badge    = document.getElementById('cvModelBadge');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn)  stopBtn.disabled  = true;
    if (badge) {
        badge.textContent = CV.modelLoaded ? 'Ready' : 'Idle';
        badge.className   = 'cv-status-badge ' + (CV.modelLoaded ? 'ready' : 'idle');
    }
    setCVStatus('ready', CV.modelLoaded ? 'Siap' : 'Belum dimuat');

    const hudEl = document.getElementById('cvDetectionInfo');
    if (hudEl) hudEl.style.display = 'none';

    clearTimeout(CV.humanTimer);
    clearTimeout(CV.lightTimer);
    addLog('CV', 'Deteksi dihentikan', 'System', 'info');
    showToast('⏹️ Deteksi dihentikan', 'info');
}

function setCVStatus(status, text) {
    const el = document.getElementById('cvSystemStatus');
    if (!el) return;
    el.textContent = text || status;
    el.className   = 'status-val ' + ({ ready: 'ok', loading: 'muted', error: '', inactive: 'muted' }[status] || 'muted');
}

function toggleBoundingBox(v) {
    CV.showBoxes                   = !!v;
    CV_CONFIG.ui.showBoundingBoxes = !!v;
    saveCVConfig();
    document.querySelectorAll('[id^="cvShowBoundingBox"]').forEach(el => { el.checked = !!v; });
}

function toggleDebugInfo(v) {
    CV.showDebug             = !!v;
    CV_CONFIG.ui.showDebugInfo = !!v;
    saveCVConfig();
    const hud = document.getElementById('cvDetectionInfo');
    if (hud) hud.style.display = (v && CV.detecting) ? 'flex' : 'none';
    document.querySelectorAll('[id^="cvShowDebugInfo"]').forEach(el => { el.checked = !!v; });
}

function updateCVConfig(pct) {
    const conf                    = (parseFloat(pct) || 60) / 100;
    CV.confidence                 = conf;
    CV_CONFIG.model.minConfidence = conf;
    saveCVConfig();
}

function saveCVRules() {
    try { localStorage.setItem('iotzy_cv_rules', JSON.stringify(CV.cvRules)); } catch (_) {}
    saveCVConfig();
}

function loadCVRules() {
    try {
        const s = localStorage.getItem('iotzy_cv_rules');
        if (s) {
            const r = JSON.parse(s);
            if (r.human) CV.cvRules.human = { ...CV.cvRules.human, ...r.human };
            if (r.light) CV.cvRules.light = { ...CV.cvRules.light, ...r.light };
        }
    } catch (_) {}

    try {
        const rules = automationEngine.getCVRules();
        if (rules.humanDetection.onDetect.length || rules.humanDetection.onAbsent.length) {
            CV.cvRules.human.onDetect = rules.humanDetection.onDetect;
            CV.cvRules.human.onAbsent = rules.humanDetection.onAbsent;
            CV.cvRules.human.delay    = rules.humanDetection.delay;
        }
        if (rules.lightCondition.onDark.length || rules.lightCondition.onBright.length) {
            CV.cvRules.light.onDark   = rules.lightCondition.onDark;
            CV.cvRules.light.onBright = rules.lightCondition.onBright;
            CV.cvRules.light.delay    = rules.lightCondition.delay;
        }
    } catch (_) {}
}

/* ==================== TOAST ==================== */
function showToast(msg, type = 'info') {
    const icons = {
        success: 'fa-check-circle',
        error:   'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info:    'fa-info-circle'
    };
    const cont = document.getElementById('toastContainer');
    if (!cont) return;
    const t       = document.createElement('div');
    t.className   = `toast ${type}`;
    t.innerHTML   = `
        <i class="fas ${icons[type] || icons.info} toast-icon"></i>
        <span class="toast-msg">${escHtml(msg)}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>`;
    cont.appendChild(t);
    setTimeout(() => {
        t.style.cssText = 'opacity:0;transform:translateX(16px);transition:all .3s';
        setTimeout(() => t.remove(), 300);
    }, 3500);
}

/* ==================== HELPERS ==================== */
function escHtml(s) {
    if (s == null) return '';
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
