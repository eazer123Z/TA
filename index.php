<?php
require_once __DIR__ . '/includes/auth.php';
requireLogin();
$user = getCurrentUser();
$settings = getUserSettings((int)$user['id']);
$devices = getUserDevices((int)$user['id']);
$sensors = getUserSensors((int)$user['id']);
?>
<!doctype html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>IoTzy Dashboard v4</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3"></script>
  <link rel="stylesheet" href="assets/css/dashboard.css" />
</head>
<body>
<div class="app-shell">
  <aside class="sidebar" id="sidebar">
    <h2>IoTzy</h2>
    <a class="nav-item active" onclick="switchPage('dashboard',this)">Overview</a>
    <a class="nav-item" onclick="switchPage('devices',this)">Perangkat <span id="navDeviceCount"><?=count($devices)?></span></a>
    <a class="nav-item" onclick="switchPage('sensors',this)">Sensor <span id="navSensorCount"><?=count($sensors)?></span></a>
    <a class="nav-item" onclick="switchPage('automation',this)">Automation</a>
    <a class="nav-item" onclick="switchPage('camera',this)">Camera & CV</a>
    <a class="nav-item" onclick="switchPage('analytics',this)">Logs</a>
    <a class="nav-item" onclick="switchPage('settings',this)">Settings</a>
    <a class="nav-item" href="logout.php">Logout</a>
  </aside>

  <main class="main-content">
    <header class="topbar"><h1 id="pageTitle">Overview</h1><div id="clock"></div></header>

    <section id="view-dashboard" class="view">
      <div class="grid-3">
        <div class="card"><div class="k">Devices</div><div id="statActiveDevicesVal">0</div></div>
        <div class="card"><div class="k">Sensors</div><div id="statSensorsOnlineVal">0</div></div>
        <div class="card"><div class="k">Person Count</div><div id="statPersonCount">0</div></div>
      </div>
      <div class="card">
        <h3>Quick Controls</h3>
        <div id="quickControlsContainer"></div>
      </div>
    </section>

    <section id="view-devices" class="view hidden"><div class="card"><button onclick="openAddDeviceModal()">+ Device</button><div id="devicesGrid"></div></div></section>
    <section id="view-sensors" class="view hidden"><div class="card"><button onclick="openAddSensorModal()">+ Sensor</button><div id="sensorsGrid"></div></div></section>
    <section id="view-automation" class="view hidden"><div class="card"><div id="automationGrid"></div></div></section>

    <section id="view-camera" class="view hidden">
      <div class="card">
        <button onclick="toggleCamera()">Toggle Camera</button>
        <button id="btnLoadModel" onclick="initializeCV()">Load Model</button>
        <button id="btnStartCV" onclick="startCVDetection()" disabled>Start Detection</button>
        <button id="btnStopCV" onclick="stopCVDetection()" disabled>Stop</button>
        <div id="cvSystemStatus">Idle</div>
        <div id="cameraFocusContainer" class="camera-wrap">
          <video id="cameraFocus" autoplay muted playsinline></video>
          <canvas id="cvOverlayCanvas"></canvas>
        </div>
        <div id="cvDetectionInfo" class="cv-hud" style="display:none">
          <span>Human: <b id="cvHumanCount">0</b></span>
          <span>FPS: <b id="cvFpsStatus">0 fps</b></span>
          <span>Light: <b id="cvBrightness">0</b></span>
        </div>
      </div>
      <div class="card"><h3>CV Automation</h3><div id="cvAutomationSettings"></div></div>
    </section>

    <section id="view-analytics" class="view hidden">
      <div class="card">
        <input placeholder="filter logs" oninput="filterLogs(this.value)">
        <table><tbody id="logBody"></tbody></table>
      </div>
    </section>

    <section id="view-settings" class="view hidden">
      <div class="card">
        <h3>MQTT</h3>
        <input id="mqttBroker" value="<?=htmlspecialchars($settings['mqtt_broker'] ?? '')?>" placeholder="broker">
        <input id="mqttPort" value="<?=htmlspecialchars((string)($settings['mqtt_port'] ?? '8884'))?>" placeholder="port">
        <button onclick="saveMQTTConfig()">Save</button>
      </div>
    </section>
  </main>
</div>

<div id="addDeviceModal" class="modal hidden"><div class="modal-inner"><input id="newDeviceName" placeholder="Device name"><input id="newDeviceTopicSub" placeholder="Topic Sub"><input id="newDeviceTopicPub" placeholder="Topic Pub"><button onclick="saveNewDevice()">Save</button><button onclick="closeAddDeviceModal()">Close</button></div></div>
<div id="addSensorModal" class="modal hidden"><div class="modal-inner"><input id="newSensorName" placeholder="Sensor name"><select id="newSensorType"><option value="temperature">temperature</option><option value="humidity">humidity</option><option value="brightness">brightness</option><option value="presence">presence</option></select><input id="newSensorTopic" placeholder="Topic"><input id="newSensorUnit" placeholder="Unit"><button onclick="saveNewSensor()">Save</button><button onclick="closeAddSensorModal()">Close</button></div></div>
<div id="toastContainer"></div>

<script>
const PHP_USER = <?=json_encode($user)?>;
const PHP_SETTINGS = <?=json_encode($settings)?>;
const PHP_DEVICES = <?=json_encode(array_values($devices))?>;
const PHP_SENSORS = <?=json_encode(array_values($sensors))?>;
</script>
<script src="assets/js/cv-config.js"></script>
<script src="assets/js/cv-detector.js"></script>
<script src="assets/js/light-analyzer.js"></script>
<script src="assets/js/automation-engine.js"></script>
<script src="assets/js/cv-ui.js"></script>
<script src="assets/js/app.js"></script>
</body>
</html>
