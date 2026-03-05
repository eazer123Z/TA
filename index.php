<?php
require_once __DIR__ . '/includes/auth.php';
requireLogin();

$user     = getCurrentUser();
$settings = getUserSettings($user['id']);
$devices  = getUserDevices($user['id']);
$sensors  = getUserSensors($user['id']);
?>
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>IoTzy — Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/paho-mqtt/1.0.1/mqttws31.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3"></script>
<link rel="stylesheet" href="assets/css/dashboard.css">
</head>
<body>

<!-- Loading Screen -->
<div id="appLoadingScreen" class="loading-screen">
  <div class="loading-inner">
    <div class="loading-logo"><i class="fas fa-bolt"></i></div>
    <h1 class="loading-title">IoTzy</h1>
    <div class="loading-bar"><div class="loading-fill"></div></div>
    <p class="loading-sub">Memuat sistem…</p>
  </div>
</div>

<div id="mainApp" class="app-shell opacity-0">

  <!-- ═══ SIDEBAR ═══ -->
  <aside id="sidebar" class="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-logo"><i class="fas fa-bolt"></i></div>
      <div>
        <span class="brand-name">IoTzy</span>
        <span class="brand-tag">Smart Home Dashboard</span>
      </div>
    </div>
    <nav class="sidebar-nav">
      <div class="nav-group-label">Monitor</div>
      <a href="#" onclick="switchPage('dashboard',this)" class="nav-item active" data-page="dashboard">
        <span class="nav-icon"><i class="fas fa-chart-treemap"></i></span>
        <span class="nav-label">Overview</span>
      </a>
      <a href="#" onclick="switchPage('devices',this)" class="nav-item" data-page="devices">
        <span class="nav-icon"><i class="fas fa-microchip"></i></span>
        <span class="nav-label">Perangkat</span>
        <span class="nav-badge" id="navDeviceCount"><?= count($devices) ?></span>
      </a>
      <a href="#" onclick="switchPage('sensors',this)" class="nav-item" data-page="sensors">
        <span class="nav-icon"><i class="fas fa-signal"></i></span>
        <span class="nav-label">Sensor</span>
        <span class="nav-badge" id="navSensorCount"><?= count($sensors) ?></span>
      </a>

      <div class="nav-group-label" style="margin-top:18px">Otomasi</div>
      <a href="#" onclick="switchPage('automation',this)" class="nav-item" data-page="automation">
        <span class="nav-icon"><i class="fas fa-sliders"></i></span>
        <span class="nav-label">Aturan Otomasi</span>
      </a>
      <a href="#" onclick="switchPage('camera',this)" class="nav-item" data-page="camera">
        <span class="nav-icon"><i class="fas fa-eye"></i></span>
        <span class="nav-label">Kamera & CV</span>
        <span class="nav-dot" id="cvNavDot"></span>
      </a>

      <div class="nav-group-label" style="margin-top:18px">Sistem</div>
      <a href="#" onclick="switchPage('analytics',this)" class="nav-item" data-page="analytics">
        <span class="nav-icon"><i class="fas fa-list-ul"></i></span>
        <span class="nav-label">Log Aktivitas</span>
      </a>
      <a href="#" onclick="switchPage('settings',this)" class="nav-item" data-page="settings">
        <span class="nav-icon"><i class="fas fa-gear"></i></span>
        <span class="nav-label">Pengaturan</span>
      </a>
    </nav>
    <div class="sidebar-footer">
      <div class="user-pill">
        <div class="user-avatar"><?= strtoupper(substr($user['username'], 0, 1)) ?></div>
        <div class="user-info">
          <span class="user-name"><?= htmlspecialchars($user['full_name'] ?: $user['username']) ?></span>
          <span class="user-role"><?= htmlspecialchars($user['role']) ?></span>
        </div>
        <a href="logout.php" class="logout-btn" title="Logout"><i class="fas fa-right-from-bracket"></i></a>
      </div>
      <div class="sidebar-bottom-row">
        <div class="mqtt-pill">
          <span class="mqtt-dot" id="sidebarMqttDot"></span>
          <span id="sidebarMqttText">Offline</span>
        </div>
        <button onclick="connectMQTT()" class="icon-btn" title="Hubungkan MQTT"
          style="border:1px solid var(--sb-border);background:rgba(255,255,255,0.04);color:var(--sb-text)">
          <i class="fas fa-wifi"></i>
        </button>
      </div>
    </div>
  </aside>

  <div id="overlay" onclick="toggleSidebar()" class="sidebar-overlay"></div>

  <!-- ═══ MAIN CONTENT ═══ -->
  <main class="main-content">
    <header class="topbar">
      <div class="topbar-left">
        <button onclick="toggleSidebar()" class="menu-btn"><i class="fas fa-bars"></i></button>
        <div class="breadcrumb">
          <span class="breadcrumb-app">IoTzy</span>
          <i class="fas fa-chevron-right breadcrumb-sep"></i>
          <span id="pageTitle" class="breadcrumb-page">Overview</span>
        </div>
      </div>
      <div class="topbar-right">
        <div class="mqtt-badge">
          <span class="mqtt-dot" id="mqttStatusDot"></span>
          <span id="mqttStatusText" class="mqtt-label">Disconnected</span>
        </div>
        <div style="text-align:right">
          <span id="clock" class="clock-time">00:00:00</span>
          <span id="date" class="clock-date">—</span>
        </div>
      </div>
    </header>

    <div class="page-wrapper">

      <!-- ════════════ DASHBOARD ════════════ -->
      <div id="view-dashboard" class="view">
        <!-- 5 stat cards including person count -->
        <div class="stats-grid" style="grid-template-columns:repeat(5,1fr)">
          <div class="stat-card">
            <div class="stat-icon green"><i class="fas fa-plug"></i></div>
            <div class="stat-body">
              <div class="stat-value" id="statActiveDevicesVal">0</div>
              <div class="stat-label">Perangkat Aktif</div>
            </div>
            <div class="stat-sub" id="statActiveDevicesSub">dari 0 total</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon blue"><i class="fas fa-signal"></i></div>
            <div class="stat-body">
              <div class="stat-value" id="statSensorsOnlineVal">0</div>
              <div class="stat-label">Sensor Online</div>
            </div>
            <div class="stat-sub" id="statSensorsOnlineSub">mengirim data</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon amber"><i class="fas fa-wifi"></i></div>
            <div class="stat-body">
              <div class="stat-value" id="statMqttVal">—</div>
              <div class="stat-label">MQTT Broker</div>
            </div>
            <div class="stat-sub" id="statMqttSub">tidak terhubung</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon indigo"><i class="fas fa-users"></i></div>
            <div class="stat-body">
              <div class="stat-value" id="statPersonCount">0</div>
              <div class="stat-label">Orang Terdeteksi</div>
            </div>
            <div class="stat-sub" id="statPersonPresence">Tidak ada</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon purple"><i class="fas fa-clock"></i></div>
            <div class="stat-body">
              <div class="stat-value" id="statUptimeVal">—</div>
              <div class="stat-label">Sesi Aktif</div>
            </div>
            <div class="stat-sub">sejak login</div>
          </div>
        </div>

        <div class="dashboard-main">
          <div class="camera-card">
            <div class="camera-header">
              <span class="card-title"><i class="fas fa-video" style="color:var(--a)"></i> Live Feed</span>
              <div style="display:flex;gap:8px;align-items:center">
                <span id="camTag" class="live-badge hidden"><i class="fas fa-circle"></i> LIVE</span>
                <button onclick="openCameraSelector()" class="icon-btn" title="Pilih kamera"><i class="fas fa-camera-rotate"></i></button>
                <button onclick="toggleCamera()" class="icon-btn" title="Toggle kamera"><i class="fas fa-power-off"></i></button>
              </div>
            </div>
            <div class="camera-body">
              <video id="camera" autoplay playsinline muted class="camera-video hidden"></video>
              <div id="camPlaceholder" class="camera-placeholder">
                <i class="fas fa-video-slash"></i>
                <span>Kamera Offline</span>
                <small>Klik power untuk mengaktifkan</small>
              </div>
            </div>
          </div>
          <div class="quick-card">
            <div class="quick-header">
              <span class="card-title"><i class="fas fa-bolt" style="color:var(--amber)"></i> Kontrol Cepat</span>
              <button onclick="openQuickControlSettings()" class="icon-btn" title="Pilih perangkat"><i class="fas fa-pen-to-square"></i></button>
            </div>
            <div id="quickControlsContainer" class="quick-body"></div>
          </div>
        </div>
      </div>

      <!-- ════════════ DEVICES ════════════ -->
      <div id="view-devices" class="view hidden">
        <div class="view-header">
          <div>
            <h2 class="view-title">Perangkat</h2>
            <p class="view-sub">Kelola semua perangkat IoT — kontrol manual &amp; otomasi berbasis sensor</p>
          </div>
          <div class="view-actions">
            <div class="search-box">
              <i class="fas fa-search"></i>
              <input type="text" placeholder="Cari perangkat…" oninput="filterDevices(this.value)">
            </div>
            <button onclick="openAddDeviceModal()" class="btn-primary"><i class="fas fa-plus"></i> Tambah</button>
          </div>
        </div>
        <div id="devicesGrid" class="device-grid"></div>
        <div id="emptyDevices" class="empty-state hidden">
          <i class="fas fa-microchip"></i>
          <p>Belum ada perangkat</p>
          <div class="hint">Tambahkan perangkat IoT Anda dan hubungkan via MQTT untuk kontrol otomatis</div>
          <button onclick="openAddDeviceModal()" class="btn-primary" style="margin-top:8px"><i class="fas fa-plus"></i> Tambah Perangkat</button>
        </div>
      </div>

      <!-- ════════════ SENSORS ════════════ -->
      <div id="view-sensors" class="view hidden">
        <div class="view-header">
          <div>
            <h2 class="view-title">Sensor</h2>
            <p class="view-sub">Monitoring data real-time dari semua sensor IoT Anda</p>
          </div>
          <div class="view-actions">
            <div class="search-box">
              <i class="fas fa-search"></i>
              <input type="text" placeholder="Cari sensor…" oninput="filterSensors(this.value)">
            </div>
            <button onclick="openAddSensorModal()" class="btn-primary"><i class="fas fa-plus"></i> Tambah</button>
          </div>
        </div>
        <div id="sensorsGrid" class="sensor-grid"></div>
        <div id="emptySensors" class="empty-state hidden">
          <i class="fas fa-signal"></i>
          <p>Belum ada sensor</p>
          <div class="hint">Tambahkan sensor untuk memantau kondisi ruangan secara real-time</div>
          <button onclick="openAddSensorModal()" class="btn-primary" style="margin-top:8px"><i class="fas fa-plus"></i> Tambah Sensor</button>
        </div>
      </div>

      <!-- ════════════ AUTOMATION ════════════ -->
      <div id="view-automation" class="view hidden">
        <div class="view-header">
          <div>
            <h2 class="view-title">Aturan Otomasi</h2>
            <p class="view-sub">Aturan otomatis berdasarkan nilai sensor — nyalakan/matikan perangkat sesuai kondisi</p>
          </div>
          <div class="view-actions">
            <button onclick="renderAutomationView()" class="btn-ghost small"><i class="fas fa-refresh"></i> Perbarui</button>
          </div>
        </div>
        <div id="automationGrid" class="automation-grid"></div>
        <div id="emptyAutomation" class="empty-state hidden">
          <i class="fas fa-sliders"></i>
          <p>Belum ada sensor</p>
          <div class="hint">Tambahkan sensor terlebih dahulu agar aturan otomasi tersedia</div>
          <button onclick="switchPage('sensors', document.querySelector('[data-page=sensors]'))" class="btn-primary" style="margin-top:8px">
            <i class="fas fa-plus"></i> Tambah Sensor
          </button>
        </div>
      </div>

      <!-- ════════════ CAMERA / CV ════════════ -->
      <div id="view-camera" class="view hidden">
        <div class="view-header">
          <div>
            <h2 class="view-title">Kamera &amp; Computer Vision</h2>
            <p class="view-sub">Deteksi objek real-time · Analisis cahaya · Automasi berbasis kamera</p>
          </div>
        </div>
        <div class="camera-layout">
          <!-- LEFT: camera feed + automation panel -->
          <div class="camera-main-col">
            <div class="camera-card tall">
              <div class="camera-header">
                <span class="card-title">Monitor Live</span>
                <div style="display:flex;gap:8px;align-items:center">
                  <span id="cameraFocusTag" class="live-badge hidden"><i class="fas fa-circle"></i> LIVE</span>
                  <span id="cvLoadingStatus" class="loading-badge hidden"><i class="fas fa-spinner fa-spin"></i> Memuat model…</span>
                  <button onclick="openCameraSelector()" class="icon-btn" title="Pilih kamera"><i class="fas fa-camera-rotate"></i></button>
                  <button onclick="toggleCameraFocus()" class="icon-btn" title="Nyala/Mati"><i class="fas fa-power-off"></i></button>
                </div>
              </div>
              <div class="camera-body" id="cameraFocusContainer" style="position:relative">
                <video id="cameraFocus" autoplay playsinline muted class="camera-video hidden"></video>
                <div id="cameraFocusPlaceholder" class="camera-placeholder">
                  <i class="fas fa-video-slash"></i>
                  <span>Kamera Offline</span>
                  <small>Klik tombol power untuk mengaktifkan</small>
                </div>
                <canvas id="cvOverlayCanvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10"></canvas>
                <!-- Person count overlay -->
                <div id="cvDetectionInfo" class="cv-hud" style="display:none">
                  <div class="hud-item hud-person-count">
                    <i class="fas fa-user"></i>
                    <span id="cvHumanCount">0</span>
                    <span style="font-size:10px;opacity:.8"> orang</span>
                  </div>
                  <div class="hud-item"><i class="fas fa-sun"></i> <span id="cvBrightness">—</span></div>
                  <div class="hud-item"><i class="fas fa-crosshairs"></i> <span id="cvConfidence">—</span></div>
                  <div class="hud-item"><i class="fas fa-tachometer-alt"></i> <span id="cvFpsStatus">—</span></div>
                </div>
              </div>
            </div>

            <!-- CV Automation Panel -->
            <div class="card">
              <div class="card-header">
                <span class="card-title"><i class="fas fa-robot" style="color:var(--a)"></i> Automasi CV</span>
                <button onclick="if(typeof cvUI!=='undefined')cvUI.renderAutomationSettings()" class="icon-btn" title="Refresh"><i class="fas fa-refresh"></i></button>
              </div>
              <div id="cvAutomationSettings" class="card-body">
                <p style="color:var(--ink-4);font-size:12px;text-align:center;padding:24px">
                  <i class="fas fa-brain" style="display:block;font-size:24px;margin-bottom:10px;opacity:.3"></i>
                  Load model CV terlebih dahulu untuk mengaktifkan automasi
                </p>
              </div>
            </div>
          </div>

          <!-- RIGHT: status + controls -->
          <div class="camera-side-col">
            <!-- Person Count prominent display -->
            <div class="card">
              <div class="card-header">
                <span class="card-title"><i class="fas fa-users" style="color:var(--a)"></i> Jumlah Orang</span>
                <span id="cvModelBadge" class="cv-status-badge idle">Idle</span>
              </div>
              <div class="card-body">
                <div class="person-count-display">
                  <div class="person-count-big" id="cvPersonCountBig">0</div>
                  <div class="person-count-label">ORANG TERDETEKSI</div>
                </div>
                <div class="status-list">
                  <div class="status-row"><span class="status-key">Status Model</span><span id="cvSystemStatus" class="status-val muted">Belum dimuat</span></div>
                  <div class="status-row"><span class="status-key">Deteksi</span><span id="cvPresenceStatus" class="status-val muted">—</span></div>
                  <div class="status-row"><span class="status-key">Cahaya</span><span id="cvLightCondition" class="status-val muted">—</span></div>
                  <div class="status-row"><span class="status-key">FPS</span><span id="cvFpsStatus2" class="status-val muted">—</span></div>
                </div>
              </div>
            </div>

            <!-- Brightness -->
            <div class="card">
              <div class="card-header"><span class="card-title"><i class="fas fa-sun" style="color:var(--amber)"></i> Kecerahan</span></div>
              <div class="card-body">
                <div class="brightness-display"><span id="cvBrightnessLabel" class="brightness-val">—</span></div>
                <div class="progress-track"><div id="cvBrightnessBar" class="progress-fill brightness-bar" style="width:0%"></div></div>
                <div class="progress-labels"><span>Gelap</span><span>Terang</span></div>
              </div>
            </div>

            <!-- Controls -->
            <div class="card">
              <div class="card-header"><span class="card-title"><i class="fas fa-gamepad" style="color:var(--purple)"></i> Kontrol CV</span></div>
              <div class="card-body cv-controls">
                <button id="btnLoadModel" onclick="initializeCV()" class="btn-cv blue">
                  <i class="fas fa-brain"></i> Load Model COCO-SSD
                </button>
                <button id="btnStartCV" onclick="startCVDetection()" class="btn-cv green" disabled>
                  <i class="fas fa-play"></i> Mulai Deteksi
                </button>
                <button id="btnStopCV" onclick="stopCVDetection()" class="btn-cv red" disabled>
                  <i class="fas fa-stop"></i> Stop Deteksi
                </button>
              </div>
            </div>

            <!-- CV Settings -->
            <div class="card">
              <div class="card-header"><span class="card-title">Pengaturan CV</span></div>
              <div class="card-body settings-body">
                <div class="setting-toggle-row">
                  <div><div class="setting-label">Bounding Box</div></div>
                  <label class="toggle-wrapper">
                    <input type="checkbox" id="cvShowBoundingBoxCamera" checked
                      onchange="toggleBoundingBox(this.checked)" class="toggle-input">
                    <span class="toggle-track"></span>
                  </label>
                </div>
                <div class="setting-toggle-row">
                  <div><div class="setting-label">Debug Overlay</div></div>
                  <label class="toggle-wrapper">
                    <input type="checkbox" id="cvShowDebugInfoCamera" checked
                      onchange="toggleDebugInfo(this.checked)" class="toggle-input">
                    <span class="toggle-track"></span>
                  </label>
                </div>
                <div class="field-group">
                  <label>Min. Confidence</label>
                  <div class="field-input-group">
                    <input type="number" id="cvConfidenceThreshold" value="60" min="10" max="99"
                      class="field-input" onchange="updateCVConfig(this.value)">
                    <span class="field-unit">%</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Capabilities -->
            <div class="card" style="background:var(--surface-2)">
              <div class="card-body">
                <div class="info-items">
                  <div class="info-item"><i class="fas fa-check-circle"></i> TensorFlow.js COCO-SSD</div>
                  <div class="info-item"><i class="fas fa-check-circle"></i> Hitung jumlah orang real-time</div>
                  <div class="info-item"><i class="fas fa-check-circle"></i> Analisis kecerahan kamera</div>
                  <div class="info-item"><i class="fas fa-check-circle"></i> Automasi 80+ kelas objek</div>
                  <div class="info-item"><i class="fas fa-check-circle"></i> Kontrol perangkat otomatis</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ════════════ ANALYTICS ════════════ -->
      <div id="view-analytics" class="view hidden">
        <div class="view-header">
          <div>
            <h2 class="view-title">Log Aktivitas</h2>
            <p class="view-sub">Riwayat semua aksi sistem, otomasi, dan MQTT</p>
          </div>
          <div class="view-actions">
            <div class="search-box">
              <i class="fas fa-search"></i>
              <input type="text" placeholder="Filter log…" oninput="filterLogs(this.value)">
            </div>
            <div class="log-filter-btns">
              <button class="log-filter-btn active" onclick="filterLogType('all'); this.closest('.log-filter-btns').querySelectorAll('.log-filter-btn').forEach(b=>b.classList.remove('active')); this.classList.add('active')">Semua</button>
              <button class="log-filter-btn" onclick="filterLogType('success'); this.closest('.log-filter-btns').querySelectorAll('.log-filter-btn').forEach(b=>b.classList.remove('active')); this.classList.add('active')">Sukses</button>
              <button class="log-filter-btn" onclick="filterLogType('error'); this.closest('.log-filter-btns').querySelectorAll('.log-filter-btn').forEach(b=>b.classList.remove('active')); this.classList.add('active')">Error</button>
            </div>
            <button onclick="exportLogsToExcel()" class="btn-ghost"><i class="fas fa-download"></i> Ekspor</button>
            <button onclick="clearLogs()" class="btn-ghost red"><i class="fas fa-trash"></i> Hapus</button>
          </div>
        </div>
        <div class="log-table-wrapper">
          <table class="log-table">
            <thead><tr><th>Tanggal</th><th>Waktu</th><th>Perangkat</th><th>Aktivitas</th><th>Trigger</th></tr></thead>
            <tbody id="logBody"></tbody>
          </table>
          <div id="emptyLog" class="empty-state hidden"><i class="fas fa-inbox"></i><p>Belum ada aktivitas tercatat</p></div>
        </div>
      </div>

      <!-- ════════════ SETTINGS ════════════ -->
      <div id="view-settings" class="view hidden">
        <div class="view-header">
          <div><h2 class="view-title">Pengaturan</h2><p class="view-sub">Konfigurasi akun, MQTT, dan sistem</p></div>
        </div>
        <div class="settings-grid">
          <div class="card">
            <div class="card-header"><span class="card-title"><i class="fas fa-user"></i> Profil Akun</span></div>
            <div class="card-body settings-body">
              <div class="field-group">
                <label>Nama Lengkap</label>
                <input type="text" id="settingFullName" class="form-input"
                  value="<?= htmlspecialchars($user['full_name'] ?? '') ?>" placeholder="Nama lengkap Anda">
              </div>
              <div class="field-group">
                <label>Email</label>
                <input type="email" id="settingEmail" class="form-input" value="<?= htmlspecialchars($user['email']) ?>">
              </div>
              <div class="field-group">
                <label>Username</label>
                <input type="text" class="form-input" value="<?= htmlspecialchars($user['username']) ?>" disabled>
              </div>
              <button onclick="saveProfile()" class="btn-primary" style="width:100%;justify-content:center">
                <i class="fas fa-floppy-disk"></i> Simpan Profil
              </button>
            </div>
          </div>

          <div class="card">
            <div class="card-header"><span class="card-title"><i class="fas fa-key"></i> Ganti Password</span></div>
            <div class="card-body settings-body">
              <div class="field-group">
                <label>Password Saat Ini</label>
                <input type="password" id="oldPassword" class="form-input" placeholder="Password lama">
              </div>
              <div class="field-group">
                <label>Password Baru</label>
                <input type="password" id="newPassword" class="form-input" placeholder="Minimal 8 karakter">
              </div>
              <div class="field-group">
                <label>Konfirmasi Password Baru</label>
                <input type="password" id="confirmPassword" class="form-input" placeholder="Ulangi password baru">
              </div>
              <button onclick="changePassword()" class="btn-primary" style="width:100%;justify-content:center">
                <i class="fas fa-lock"></i> Ganti Password
              </button>
            </div>
          </div>

          <div class="card">
            <div class="card-header"><span class="card-title"><i class="fas fa-network-wired"></i> Koneksi MQTT</span></div>
            <div class="card-body settings-body">
              <div class="setting-row">
                <div>
                  <div class="setting-row-label">Status Broker</div>
                  <div id="mqttStatusSettings" class="setting-val muted">Disconnected</div>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap">
                  <button onclick="connectMQTT()" class="btn-primary small">Hubungkan</button>
                  <button onclick="disconnectMQTT()" class="btn-ghost small">Putus</button>
                </div>
              </div>
              <button onclick="openMQTTConfigModal()" class="btn-ghost full"><i class="fas fa-gear"></i> Konfigurasi Broker</button>
            </div>
          </div>

          <div class="card">
            <div class="card-header"><span class="card-title"><i class="fas fa-brain"></i> Computer Vision</span></div>
            <div class="card-body settings-body">
              <div class="setting-toggle-row">
                <div><div class="setting-row-label">Bounding Box</div><div class="setting-hint">Kotak deteksi di sekitar objek</div></div>
                <label class="toggle-wrapper">
                  <input type="checkbox" id="cvShowBoundingBoxSettings" checked
                    onchange="toggleBoundingBox(this.checked)" class="toggle-input">
                  <span class="toggle-track"></span>
                </label>
              </div>
              <div class="setting-toggle-row">
                <div><div class="setting-row-label">Debug Overlay</div><div class="setting-hint">Informasi deteksi di layar kamera</div></div>
                <label class="toggle-wrapper">
                  <input type="checkbox" id="cvShowDebugInfoSettings" checked
                    onchange="toggleDebugInfo(this.checked)" class="toggle-input">
                  <span class="toggle-track"></span>
                </label>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="card-header"><span class="card-title"><i class="fas fa-circle-info"></i> Info Aplikasi</span></div>
            <div class="card-body">
              <div class="info-table">
                <div class="info-row"><span>Versi</span><span class="mono">4.0.0</span></div>
                <div class="info-row"><span>Perangkat</span><span class="mono" id="totalDevices"><?= count($devices) ?></span></div>
                <div class="info-row"><span>Sensor</span><span class="mono" id="totalSensors"><?= count($sensors) ?></span></div>
                <div class="info-row"><span>Login sebagai</span><span class="mono"><?= htmlspecialchars($user['username']) ?></span></div>
                <div class="info-row"><span>Role</span><span class="mono"><?= $user['role'] ?></span></div>
              </div>
            </div>
          </div>

          <div class="card" style="border-color:rgba(220,38,38,0.2)">
            <div class="card-header"><span class="card-title" style="color:var(--red)"><i class="fas fa-triangle-exclamation"></i> Zona Berbahaya</span></div>
            <div class="card-body settings-body">
              <p class="setting-hint" style="margin-bottom:16px">Tindakan ini tidak dapat dibatalkan. Harap berhati-hati.</p>
              <div style="display:flex;gap:10px;flex-wrap:wrap">
                <button onclick="clearLogs()" class="btn-ghost red"><i class="fas fa-trash"></i> Hapus Semua Log</button>
                <a href="logout.php" class="btn-danger"><i class="fas fa-right-from-bracket"></i> Logout</a>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div><!-- /page-wrapper -->
  </main>
</div><!-- /app-shell -->

<!-- ══════════════ MODALS ══════════════ -->

<!-- Quick Control Modal -->
<div id="quickControlModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-header"><h3>Pilih Kontrol Cepat</h3><button onclick="closeQuickControlSettings()" class="modal-close"><i class="fas fa-times"></i></button></div>
    <p class="modal-sub" style="padding:0 20px 8px">Pilih hingga 4 perangkat untuk tampil di dashboard</p>
    <div id="quickControlDevicesList" class="modal-list" style="padding:0 20px 12px"></div>
    <div class="modal-footer"><button onclick="closeQuickControlSettings()" class="btn-ghost">Batal</button><button onclick="saveQuickControlSettings()" class="btn-primary">Simpan</button></div>
  </div>
</div>

<!-- Camera Selector -->
<div id="cameraSelectorModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-header"><h3>Pilih Kamera</h3><button onclick="closeCameraSelector()" class="modal-close"><i class="fas fa-times"></i></button></div>
    <div id="cameraDevicesList" class="modal-list" style="padding:12px 20px"><p class="modal-loading">Memuat daftar kamera…</p></div>
    <div class="modal-footer"><button onclick="closeCameraSelector()" class="btn-ghost">Tutup</button></div>
  </div>
</div>

<!-- Device Setting Modal -->
<div id="topicModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-header">
      <h3>Setting — <span id="topicDeviceName"></span></h3>
      <button onclick="closeTopicSettings()" class="modal-close"><i class="fas fa-times"></i></button>
    </div>
    <div class="modal-fields">
      <div class="field-row-2">
        <div class="field-group">
          <label>Nama Perangkat</label>
          <input type="text" id="editDeviceName" class="form-input" placeholder="Nama perangkat">
        </div>
        <div class="field-group">
          <label>Icon Perangkat</label>
          <select id="editDeviceIcon" class="form-input form-select">
            <option value="fa-lightbulb">💡 Lampu</option>
            <option value="fa-wind">🌀 Kipas Angin</option>
            <option value="fa-snowflake">❄️ AC / Pendingin</option>
            <option value="fa-tv">📺 Televisi</option>
            <option value="fa-lock">🔒 Kunci Pintu</option>
            <option value="fa-door-open">🚪 Pintu</option>
            <option value="fa-video">📹 Kamera CCTV</option>
            <option value="fa-volume-up">🔊 Speaker</option>
            <option value="fa-plug">🔌 Stop Kontak</option>
          </select>
        </div>
      </div>
      <div class="field-group">
        <label>Subscribe Topic <span class="field-hint">(status dari device → dashboard)</span></label>
        <input type="text" id="deviceTopicSub" class="form-input" placeholder="iotzy/device/lampu/status">
      </div>
      <div class="field-group">
        <label>Publish Topic <span class="field-hint">(kontrol dashboard → device)</span></label>
        <input type="text" id="deviceTopicPub" class="form-input" placeholder="iotzy/device/lampu/control">
      </div>
      <p class="form-hint">Payload ON: <code>{"state": 1}</code> · Payload OFF: <code>{"state": 0}</code></p>
    </div>
    <div class="modal-footer"><button onclick="closeTopicSettings()" class="btn-ghost">Batal</button><button onclick="saveDeviceSettings()" class="btn-primary">Simpan</button></div>
  </div>
</div>

<!-- Add Device Modal -->
<div id="addDeviceModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-header"><h3>Tambah Perangkat Baru</h3><button onclick="closeAddDeviceModal()" class="modal-close"><i class="fas fa-times"></i></button></div>
    <div class="modal-fields">
      <div class="field-group"><label>Nama Perangkat</label><input type="text" id="newDeviceName" class="form-input" placeholder="cth: Lampu Kamar Tidur, Kipas Ruang Tamu"></div>
      <div class="field-group">
        <label>Jenis / Icon</label>
        <select id="newDeviceIcon" class="form-input form-select">
          <option value="fa-lightbulb">💡 Lampu (LED / Bohlam)</option>
          <option value="fa-wind">🌀 Kipas Angin</option>
          <option value="fa-snowflake">❄️ AC / Pendingin</option>
          <option value="fa-tv">📺 Televisi</option>
          <option value="fa-lock">🔒 Kunci Pintu Otomatis</option>
          <option value="fa-door-open">🚪 Sensor / Aktuator Pintu</option>
          <option value="fa-video">📹 Kamera CCTV</option>
          <option value="fa-volume-up">🔊 Speaker / Alarm</option>
          <option value="fa-plug">🔌 Stop Kontak Pintar</option>
        </select>
      </div>
      <div class="field-row-2">
        <div class="field-group"><label>Subscribe Topic</label><input type="text" id="newDeviceTopicSub" class="form-input" placeholder="iotzy/device/xxx/status"></div>
        <div class="field-group"><label>Publish Topic</label><input type="text" id="newDeviceTopicPub" class="form-input" placeholder="iotzy/device/xxx/control"></div>
      </div>
      <p class="form-hint">Kipas angin mendukung kontrol kecepatan via slider setelah ditambahkan.</p>
    </div>
    <div class="modal-footer"><button onclick="closeAddDeviceModal()" class="btn-ghost">Batal</button><button onclick="saveNewDevice()" class="btn-primary"><i class="fas fa-plus"></i> Tambah Perangkat</button></div>
  </div>
</div>

<!-- Add Sensor Modal -->
<div id="addSensorModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-header"><h3>Tambah Sensor Baru</h3><button onclick="closeAddSensorModal()" class="modal-close"><i class="fas fa-times"></i></button></div>
    <div class="modal-fields">
      <div class="field-group"><label>Nama Sensor</label><input type="text" id="newSensorName" class="form-input" placeholder="cth: Sensor Suhu Ruang Tamu"></div>
      <div class="field-row-2">
        <div class="field-group">
          <label>Tipe Sensor</label>
          <select id="newSensorType" class="form-input form-select">
            <option value="temperature">🌡️ Suhu (DHT11/DHT22)</option>
            <option value="humidity">💧 Kelembaban</option>
            <option value="air_quality">💨 Kualitas Udara (MQ135)</option>
            <option value="presence">👤 Kehadiran (PIR)</option>
            <option value="brightness">☀️ Kecerahan (LDR)</option>
            <option value="motion">🏃 Gerakan</option>
            <option value="smoke">🔥 Asap (MQ2)</option>
            <option value="gas">⚠️ Gas (MQ7)</option>
          </select>
        </div>
        <div class="field-group"><label>Satuan</label><input type="text" id="newSensorUnit" class="form-input" placeholder="°C, %, ppm, lux…"></div>
      </div>
      <div class="field-group"><label>MQTT Topic</label><input type="text" id="newSensorTopic" class="form-input" placeholder="iotzy/sensor/suhu"></div>
      <p class="form-hint">Payload yang diterima: <code>{"value": 28.5}</code> atau nilai langsung <code>28.5</code></p>
    </div>
    <div class="modal-footer"><button onclick="closeAddSensorModal()" class="btn-ghost">Batal</button><button onclick="saveNewSensor()" class="btn-primary"><i class="fas fa-plus"></i> Tambah Sensor</button></div>
  </div>
</div>

<!-- Sensor Setting Modal -->
<div id="sensorSettingModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-header">
      <h3>Setting — <span id="ssSensorName"></span></h3>
      <button onclick="closeSensorSettings()" class="modal-close"><i class="fas fa-times"></i></button>
    </div>
    <div class="modal-fields">
      <div class="field-group"><label>Nama Sensor</label><input type="text" id="ssEditName" class="form-input"></div>
      <div class="field-row-2">
        <div class="field-group">
          <label>Tipe</label>
          <select id="ssEditType" class="form-input form-select">
            <option value="temperature">🌡️ Suhu</option>
            <option value="humidity">💧 Kelembaban</option>
            <option value="air_quality">💨 Kualitas Udara</option>
            <option value="presence">👤 Kehadiran</option>
            <option value="brightness">☀️ Kecerahan</option>
            <option value="motion">🏃 Gerakan</option>
            <option value="smoke">🔥 Asap</option>
            <option value="gas">⚠️ Gas</option>
          </select>
        </div>
        <div class="field-group"><label>Satuan <span class="field-hint">(opsional)</span></label><input type="text" id="ssEditUnit" class="form-input" placeholder="°C, %, ppm…"></div>
      </div>
      <div class="field-group">
        <label>MQTT Topic</label>
        <input type="text" id="ssEditTopic" class="form-input">
      </div>
    </div>
    <div class="modal-footer"><button onclick="closeSensorSettings()" class="btn-ghost">Batal</button><button onclick="saveSensorSettings()" class="btn-primary">Simpan</button></div>
  </div>
</div>

<!-- MQTT Config Modal -->
<div id="mqttConfigModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-header"><h3>Konfigurasi MQTT</h3><button onclick="closeMQTTConfigModal()" class="modal-close"><i class="fas fa-times"></i></button></div>
    <div class="modal-fields">
      <div class="field-group"><label>Broker URL</label><input type="text" id="mqttBroker" value="<?= htmlspecialchars($settings['mqtt_broker']) ?>" class="form-input"></div>
      <div class="field-row-2">
        <div class="field-group"><label>Port</label><input type="number" id="mqttPort" value="<?= $settings['mqtt_port'] ?>" class="form-input"></div>
        <div class="field-group"><label>Client ID</label><input type="text" id="mqttClientId" value="<?= htmlspecialchars($settings['mqtt_client_id'] ?? 'iotzy_web') ?>" class="form-input"></div>
      </div>
      <div class="field-row-2">
        <div class="field-group"><label>Path</label><input type="text" id="mqttPath" value="<?= htmlspecialchars($settings['mqtt_path']) ?>" class="form-input"></div>
        <div class="field-group" style="justify-content:flex-end;padding-top:22px">
          <label class="toggle-label-row">
            <span>Gunakan SSL</span>
            <label class="toggle-wrapper">
              <input type="checkbox" id="mqttUseSSL" <?= $settings['mqtt_use_ssl'] ? 'checked' : '' ?> class="toggle-input">
              <span class="toggle-track"></span>
            </label>
          </label>
        </div>
      </div>
      <div class="field-row-2">
        <div class="field-group"><label>Username <span class="field-hint">(opsional)</span></label><input type="text" id="mqttUsername" value="<?= htmlspecialchars($settings['mqtt_username'] ?? '') ?>" class="form-input"></div>
        <div class="field-group"><label>Password <span class="field-hint">(opsional)</span></label><input type="password" id="mqttPassword" class="form-input" placeholder="••••••••"></div>
      </div>
    </div>
    <div class="modal-footer"><button onclick="closeMQTTConfigModal()" class="btn-ghost">Batal</button><button onclick="saveMQTTConfig()" class="btn-primary">Simpan &amp; Hubungkan</button></div>
  </div>
</div>

<!-- Add Automation Rule Modal -->
<div id="addRuleModal" class="modal-backdrop">
  <div class="modal">
    <div class="modal-header">
      <h3>Tambah Aturan Otomasi</h3>
      <button onclick="closeAddRuleModal()" class="modal-close"><i class="fas fa-times"></i></button>
    </div>
    <div class="modal-fields">
      <div class="field-group" style="flex-direction:row;align-items:center;gap:10px;background:var(--surface-2);padding:10px 12px;border-radius:var(--r-md);border:1px solid var(--border)">
        <div id="addRuleSensorIcon"></div>
        <div>
          <div style="font-size:10px;font-weight:700;color:var(--ink-4);text-transform:uppercase;letter-spacing:.5px">Sensor</div>
          <div id="addRuleSensorLabel" style="font-size:13px;font-weight:600;color:var(--ink)">—</div>
        </div>
      </div>
      <div class="field-group">
        <label>Kondisi Trigger</label>
        <select id="addRuleCondition" class="form-input form-select"></select>
      </div>
      <div id="addRuleThresholdRow" class="field-group">
        <label>Nilai Ambang</label>
        <div class="field-input-group">
          <input type="number" id="addRuleThreshold" class="field-input" step="0.1">
          <span class="field-unit" id="addRuleUnit">°C</span>
        </div>
      </div>
      <div id="addRuleRangeRow" class="field-row-2" style="display:none">
        <div class="field-group">
          <label>Batas Bawah</label>
          <input type="number" id="addRuleThresholdMin" class="form-input" step="0.1">
        </div>
        <div class="field-group">
          <label>Batas Atas</label>
          <input type="number" id="addRuleThresholdMax" class="form-input" step="0.1">
        </div>
      </div>
      <div class="field-group">
        <label>Perangkat yang Dikontrol</label>
        <select id="addRuleDevice" class="form-input form-select"></select>
      </div>
      <div class="field-row-2">
        <div class="field-group">
          <label>Aksi</label>
          <select id="addRuleAction" class="form-input form-select">
            <option value="on">⚡ Nyalakan (ON)</option>
            <option value="off">✕ Matikan (OFF)</option>
          </select>
        </div>
        <div class="field-group">
          <label>Delay <span class="field-hint">(ms, 0=langsung)</span></label>
          <input type="number" id="addRuleDelay" class="form-input" value="0" min="0" step="500">
        </div>
      </div>
    </div>
    <div class="modal-footer">
      <button onclick="closeAddRuleModal()" class="btn-ghost">Batal</button>
      <button onclick="saveNewAutomationRule()" class="btn-primary"><i class="fas fa-plus"></i> Tambah Aturan</button>
    </div>
  </div>
</div>

<!-- Toast container -->
<div id="toastContainer"></div>

<!-- PHP data injection -->
<script>
const PHP_USER     = <?= json_encode($user) ?>;
const PHP_SETTINGS = <?= json_encode($settings) ?>;
const PHP_DEVICES  = <?= json_encode(array_values($devices)) ?>;
const PHP_SENSORS  = <?= json_encode(array_values($sensors)) ?>;
</script>

<!-- Script load order matters! -->
<script src="assets/js/cv-config.js"></script>
<script src="assets/js/cv-detector.js"></script>
<script src="assets/js/light-analyzer.js"></script>
<script src="assets/js/automation-engine.js"></script>
<script src="assets/js/cv-ui.js"></script>
<script src="assets/js/app.js"></script>

</body>
</html>
