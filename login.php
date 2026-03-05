<?php
require_once __DIR__ . '/includes/auth.php';

startSecureSession();
if (isLoggedIn()) {
    header('Location: index.php');
    exit;
}

$error = '';
$success = '';
$mode = $_GET['mode'] ?? 'login'; // login | register
if (!in_array($mode, ['login', 'register'], true)) {
    $mode = 'login';
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $action = $_POST['action'] ?? 'login';
    $csrfToken = (string)($_POST['csrf_token'] ?? '');
    if (!verifyCsrfToken($csrfToken)) {
        $error = 'Sesi form tidak valid. Muat ulang halaman lalu coba lagi.';
        $action = '';
    }

    if ($action === 'login') {
        $username = trim($_POST['username'] ?? '');
        $password = $_POST['password'] ?? '';

        if (empty($username) || empty($password)) {
            $error = 'Semua field wajib diisi.';
        } else {
            $result = loginUser($username, $password);
            if ($result['success']) {
                header('Location: index.php');
                exit;
            } else {
                $error = $result['message'];
            }
        }
    } elseif ($action === 'register') {
        $username  = trim($_POST['username'] ?? '');
        $email     = trim($_POST['email'] ?? '');
        $password  = $_POST['password'] ?? '';
        $password2 = $_POST['password2'] ?? '';
        $fullName  = trim($_POST['full_name'] ?? '');

        if ($password !== $password2) {
            $error = 'Konfirmasi password tidak cocok.';
            $mode = 'register';
        } else {
            $result = registerUser($username, $email, $password, $fullName);
            if ($result['success']) {
                $success = $result['message'];
                $mode = 'login';
            } else {
                $error = $result['message'];
                $mode = 'register';
            }
        }
    }
}
?>
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<title>IoTzy — <?= $mode === 'register' ? 'Daftar Akun' : 'Masuk' ?></title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@400;500&family=Sora:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --ink: #0a0a0a;
  --ink-2: #3a3a3a;
  --ink-3: #888;
  --paper: #fafafa;
  --paper-2: #f0f0f0;
  --border: #e0e0e0;
  --border-2: #ccc;
  --white: #ffffff;
  --error: #c0392b;
  --success-col: #1a7a3f;
  --r: 10px;
  --t: 180ms ease;
}

html, body {
  height: 100%;
  background: var(--paper);
  font-family: 'Sora', sans-serif;
  color: var(--ink);
}

body {
  display: grid;
  grid-template-columns: 1fr 1fr;
  min-height: 100vh;
}

/* LEFT PANEL */
.panel-left {
  background: var(--ink);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 48px;
  position: relative;
  overflow: hidden;
}

.panel-left::before {
  content: '';
  position: absolute;
  top: -120px;
  right: -80px;
  width: 400px;
  height: 400px;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.06);
}
.panel-left::after {
  content: '';
  position: absolute;
  bottom: -80px;
  left: -60px;
  width: 300px;
  height: 300px;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.04);
}

.left-logo {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-mark {
  width: 40px;
  height: 40px;
  background: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  color: var(--ink);
  flex-shrink: 0;
}

.logo-text {
  font-family: 'Instrument Serif', serif;
  font-size: 26px;
  color: white;
  letter-spacing: -0.5px;
}

.left-middle {
  position: relative;
  z-index: 1;
}

.left-tagline {
  font-family: 'Instrument Serif', serif;
  font-size: 46px;
  line-height: 1.1;
  color: white;
  letter-spacing: -1.5px;
  margin-bottom: 20px;
}

.left-tagline em {
  font-style: italic;
  color: rgba(255,255,255,0.5);
}

.left-desc {
  font-size: 14px;
  color: rgba(255,255,255,0.45);
  line-height: 1.7;
  max-width: 360px;
  font-weight: 300;
}

.left-features {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-top: 32px;
}

.feature-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 40px;
  font-size: 12px;
  color: rgba(255,255,255,0.6);
  width: fit-content;
}

.feature-pill i { font-size: 11px; color: rgba(255,255,255,0.4); }

.left-bottom {
  font-size: 12px;
  color: rgba(255,255,255,0.25);
  font-family: 'DM Mono', monospace;
}

/* RIGHT PANEL */
.panel-right {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px 64px;
  background: var(--white);
  position: relative;
}

.auth-box {
  width: 100%;
  max-width: 400px;
}

.auth-header {
  margin-bottom: 36px;
}

.auth-title {
  font-family: 'Instrument Serif', serif;
  font-size: 34px;
  letter-spacing: -1px;
  color: var(--ink);
  margin-bottom: 6px;
}

.auth-sub {
  font-size: 13px;
  color: var(--ink-3);
  font-weight: 300;
}

/* Form */
.form-group {
  margin-bottom: 18px;
}

.form-label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  color: var(--ink-2);
  margin-bottom: 7px;
  letter-spacing: 0.3px;
  text-transform: uppercase;
}

.input-wrap {
  position: relative;
}

.input-icon {
  position: absolute;
  left: 14px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--ink-3);
  font-size: 13px;
  pointer-events: none;
}

.form-input {
  width: 100%;
  padding: 12px 14px 12px 40px;
  background: var(--paper);
  border: 1.5px solid var(--border);
  border-radius: var(--r);
  font-size: 14px;
  font-family: 'Sora', sans-serif;
  color: var(--ink);
  outline: none;
  transition: border-color var(--t), box-shadow var(--t);
  -webkit-appearance: none;
}

.form-input:focus {
  border-color: var(--ink);
  background: var(--white);
  box-shadow: 0 0 0 3px rgba(10,10,10,0.06);
}

.form-input::placeholder { color: var(--ink-3); font-weight: 300; }

.form-row-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

/* Alerts */
.alert {
  padding: 12px 16px;
  border-radius: var(--r);
  font-size: 13px;
  margin-bottom: 20px;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  font-weight: 500;
}

.alert-error {
  background: #fff0f0;
  border: 1px solid #f5c6c6;
  color: var(--error);
}

.alert-success {
  background: #f0fff6;
  border: 1px solid #b8e8cc;
  color: var(--success-col);
}

.alert i { margin-top: 1px; flex-shrink: 0; }

/* Submit button */
.btn-submit {
  width: 100%;
  padding: 14px;
  background: var(--ink);
  color: white;
  border: none;
  border-radius: var(--r);
  font-size: 14px;
  font-weight: 600;
  font-family: 'Sora', sans-serif;
  cursor: pointer;
  transition: all var(--t);
  letter-spacing: 0.3px;
  margin-top: 6px;
}

.btn-submit:hover {
  background: #222;
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(0,0,0,0.12);
}

.btn-submit:active { transform: translateY(0); }

/* Switch mode */
.auth-switch {
  text-align: center;
  margin-top: 28px;
  font-size: 13px;
  color: var(--ink-3);
}

.auth-switch a {
  color: var(--ink);
  font-weight: 600;
  text-decoration: none;
  border-bottom: 1px solid var(--ink);
  padding-bottom: 1px;
  transition: opacity var(--t);
}

.auth-switch a:hover { opacity: 0.6; }

/* Demo creds */
.demo-creds {
  margin-top: 24px;
  padding: 14px 16px;
  background: var(--paper);
  border: 1px solid var(--border);
  border-radius: var(--r);
  font-size: 12px;
  color: var(--ink-3);
  font-family: 'DM Mono', monospace;
}

.demo-creds strong { color: var(--ink-2); display: block; margin-bottom: 6px; font-family: 'Sora', sans-serif; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.demo-row { display: flex; justify-content: space-between; padding: 3px 0; }

/* Divider */
.divider {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 22px 0;
  color: var(--ink-3);
  font-size: 12px;
}
.divider::before, .divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* Responsive */
@media (max-width: 800px) {
  body { grid-template-columns: 1fr; }
  .panel-left { display: none; }
  .panel-right { padding: 32px 28px; }
}

/* Password toggle */
.pw-toggle {
  position: absolute;
  right: 14px;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  cursor: pointer;
  color: var(--ink-3);
  font-size: 13px;
  padding: 4px;
  transition: color var(--t);
}
.pw-toggle:hover { color: var(--ink); }
.form-input.has-toggle { padding-right: 40px; }
</style>
</head>
<body>

<!-- Left Panel -->
<div class="panel-left">
  <div class="left-logo">
    <div class="logo-mark"><i class="fas fa-bolt"></i></div>
    <span class="logo-text">IoTzy</span>
  </div>

  <div class="left-middle">
    <h1 class="left-tagline">Smart Room<br><em>Dashboard</em><br>Control.</h1>
    <p class="left-desc">Pantau dan kendalikan perangkat IoT Anda secara real-time dengan deteksi kecerdasan buatan dan otomasi berbasis sensor.</p>
    <div class="left-features">
      <div class="feature-pill"><i class="fas fa-wifi"></i> MQTT Real-time</div>
      <div class="feature-pill"><i class="fas fa-brain"></i> Computer Vision (COCO-SSD)</div>
      <div class="feature-pill"><i class="fas fa-sliders"></i> Automation Engine</div>
      <div class="feature-pill"><i class="fas fa-chart-line"></i> Activity Logging</div>
    </div>
  </div>

  <div class="left-bottom">v2.0 · IoTzy CV Dashboard · <?= date('Y') ?></div>
</div>

<!-- Right Panel -->
<div class="panel-right">
  <div class="auth-box">

    <?php if ($mode === 'login'): ?>
    <!-- LOGIN FORM -->
    <div class="auth-header">
      <h2 class="auth-title">Selamat Datang</h2>
      <p class="auth-sub">Masuk ke akun Anda untuk melanjutkan</p>
    </div>

    <?php if ($error): ?>
    <div class="alert alert-error"><i class="fas fa-circle-xmark"></i> <?= htmlspecialchars($error) ?></div>
    <?php endif; ?>
    <?php if ($success): ?>
    <div class="alert alert-success"><i class="fas fa-check-circle"></i> <?= htmlspecialchars($success) ?></div>
    <?php endif; ?>

    <form method="POST" autocomplete="on">
      <input type="hidden" name="action" value="login">
      <input type="hidden" name="csrf_token" value="<?= htmlspecialchars(getCsrfToken()) ?>">

      <div class="form-group">
        <label class="form-label">Username atau Email</label>
        <div class="input-wrap">
          <i class="fas fa-user input-icon"></i>
          <input type="text" name="username" class="form-input" placeholder="username / email" autocomplete="username" value="<?= htmlspecialchars($_POST['username'] ?? '') ?>" required>
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">Password</label>
        <div class="input-wrap">
          <i class="fas fa-lock input-icon"></i>
          <input type="password" name="password" id="pw1" class="form-input has-toggle" placeholder="••••••••" autocomplete="current-password" required>
          <button type="button" class="pw-toggle" onclick="togglePw('pw1',this)"><i class="fas fa-eye"></i></button>
        </div>
      </div>

      <button type="submit" class="btn-submit">Masuk <i class="fas fa-arrow-right" style="margin-left:6px"></i></button>
    </form>

    <div class="demo-creds">
      <strong>Demo Credentials</strong>
      <div class="demo-row"><span>Admin:</span><span>admin / Admin@123</span></div>
      <div class="demo-row"><span>User:</span><span>demo / User@123</span></div>
    </div>

    <div class="auth-switch">Belum punya akun? <a href="?mode=register">Daftar sekarang</a></div>

    <?php else: ?>
    <!-- REGISTER FORM -->
    <div class="auth-header">
      <h2 class="auth-title">Buat Akun</h2>
      <p class="auth-sub">Daftarkan akun baru untuk mengakses dashboard</p>
    </div>

    <?php if ($error): ?>
    <div class="alert alert-error"><i class="fas fa-circle-xmark"></i> <?= htmlspecialchars($error) ?></div>
    <?php endif; ?>

    <form method="POST" autocomplete="off">
      <input type="hidden" name="action" value="register">
      <input type="hidden" name="csrf_token" value="<?= htmlspecialchars(getCsrfToken()) ?>">

      <div class="form-group">
        <label class="form-label">Nama Lengkap</label>
        <div class="input-wrap">
          <i class="fas fa-id-card input-icon"></i>
          <input type="text" name="full_name" class="form-input" placeholder="Nama lengkap Anda" value="<?= htmlspecialchars($_POST['full_name'] ?? '') ?>">
        </div>
      </div>

      <div class="form-row-2">
        <div class="form-group">
          <label class="form-label">Username</label>
          <div class="input-wrap">
            <i class="fas fa-at input-icon"></i>
            <input type="text" name="username" class="form-input" placeholder="username" value="<?= htmlspecialchars($_POST['username'] ?? '') ?>" required>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Email</label>
          <div class="input-wrap">
            <i class="fas fa-envelope input-icon"></i>
            <input type="email" name="email" class="form-input" placeholder="email@domain.com" value="<?= htmlspecialchars($_POST['email'] ?? '') ?>" required>
          </div>
        </div>
      </div>

      <div class="form-row-2">
        <div class="form-group">
          <label class="form-label">Password</label>
          <div class="input-wrap">
            <i class="fas fa-lock input-icon"></i>
            <input type="password" name="password" id="pw2" class="form-input has-toggle" placeholder="Min 8 karakter" required>
            <button type="button" class="pw-toggle" onclick="togglePw('pw2',this)"><i class="fas fa-eye"></i></button>
          </div>
        </div>
        <div class="form-group">
          <label class="form-label">Konfirmasi</label>
          <div class="input-wrap">
            <i class="fas fa-lock input-icon"></i>
            <input type="password" name="password2" id="pw3" class="form-input has-toggle" placeholder="Ulangi password" required>
            <button type="button" class="pw-toggle" onclick="togglePw('pw3',this)"><i class="fas fa-eye"></i></button>
          </div>
        </div>
      </div>

      <button type="submit" class="btn-submit">Buat Akun <i class="fas fa-arrow-right" style="margin-left:6px"></i></button>
    </form>

    <div class="auth-switch">Sudah punya akun? <a href="?mode=login">Masuk di sini</a></div>
    <?php endif; ?>

  </div>
</div>

<script>
function togglePw(id, btn) {
  const input = document.getElementById(id);
  const icon = btn.querySelector('i');
  if (input.type === 'password') {
    input.type = 'text';
    icon.className = 'fas fa-eye-slash';
  } else {
    input.type = 'password';
    icon.className = 'fas fa-eye';
  }
}
</script>
</body>
</html>
