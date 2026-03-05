<?php
require_once __DIR__ . '/../config.php';

// ==================== SESSION MANAGEMENT ====================
function startSecureSession(): void {
    if (session_status() === PHP_SESSION_NONE) {
        $isHttps = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off')
            || (int)($_SERVER['SERVER_PORT'] ?? 0) === 443;
        session_set_cookie_params([
            'lifetime' => SESSION_LIFETIME,
            'path'     => '/',
            'secure'   => $isHttps,
            'httponly' => true,
            'samesite' => 'Strict'
        ]);
        session_start();

        if (!isset($_SESSION['csrf_token'])) {
            $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
        }
    }
}

function getCsrfToken(): string {
    startSecureSession();
    return (string)$_SESSION['csrf_token'];
}

function verifyCsrfToken(?string $token): bool {
    startSecureSession();
    if (!is_string($token) || $token === '') {
        return false;
    }
    return hash_equals((string)$_SESSION['csrf_token'], $token);
}

function isLoggedIn(): bool {
    startSecureSession();
    return isset($_SESSION['user_id']) && isset($_SESSION['token']);
}

function requireLogin(): void {
    if (!isLoggedIn()) {
        header('Location: login.php');
        exit;
    }
}

function getCurrentUser(): ?array {
    if (!isLoggedIn()) return null;
    $db = getDB();
    $stmt = $db->prepare("SELECT id, username, email, full_name, role, last_login FROM users WHERE id = ? AND is_active = 1");
    $stmt->execute([$_SESSION['user_id']]);
    return $stmt->fetch() ?: null;
}

// ==================== AUTH FUNCTIONS ====================
function loginUser(string $username, string $password): array {
    $db = getDB();

    $username = trim($username);
    if ($username === '' || $password === '') {
        return ['success' => false, 'message' => 'Username dan password wajib diisi.'];
    }

    // Find user by username or email
    $stmt = $db->prepare("SELECT * FROM users WHERE (username = ? OR email = ?) AND is_active = 1 LIMIT 1");
    $stmt->execute([$username, $username]);
    $user = $stmt->fetch();

    if (!$user || !password_verify($password, $user['password_hash'])) {
        return ['success' => false, 'message' => 'Username atau password salah.'];
    }

    // Start session
    startSecureSession();
    session_regenerate_id(true);

    $token = bin2hex(random_bytes(32));
    $_SESSION['user_id'] = $user['id'];
    $_SESSION['username'] = $user['username'];
    $_SESSION['role'] = $user['role'];
    $_SESSION['token'] = $token;

    // Update last login
    $db->prepare("UPDATE users SET last_login = NOW() WHERE id = ?")->execute([$user['id']]);

    // Create settings if not exists
    $db->prepare("INSERT IGNORE INTO user_settings (user_id) VALUES (?)")->execute([$user['id']]);

    return ['success' => true, 'message' => 'Login berhasil!', 'user' => $user];
}

function logoutUser(): void {
    startSecureSession();
    $_SESSION = [];
    if (ini_get("session.use_cookies")) {
        $params = session_get_cookie_params();
        setcookie(session_name(), '', time() - 42000,
            $params["path"], $params["domain"],
            $params["secure"], $params["httponly"]
        );
    }
    session_destroy();
}

function registerUser(string $username, string $email, string $password, string $fullName): array {
    $username = trim($username);
    $email = strtolower(trim($email));
    $fullName = trim($fullName);

    if (strlen($username) < 3 || strlen($username) > 50) {
        return ['success' => false, 'message' => 'Username harus 3-50 karakter.'];
    }
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        return ['success' => false, 'message' => 'Format email tidak valid.'];
    }
    if (strlen($password) < 8) {
        return ['success' => false, 'message' => 'Password minimal 8 karakter.'];
    }
    if (!preg_match('/^[a-zA-Z0-9_]+$/', $username)) {
        return ['success' => false, 'message' => 'Username hanya boleh huruf, angka, underscore.'];
    }

    $db = getDB();

    // Check duplicate
    $stmt = $db->prepare("SELECT id FROM users WHERE username = ? OR email = ?");
    $stmt->execute([$username, $email]);
    if ($stmt->fetch()) {
        return ['success' => false, 'message' => 'Username atau email sudah digunakan.'];
    }

    try {
        $db->beginTransaction();

        $hash = password_hash($password, PASSWORD_BCRYPT, ['cost' => 12]);
        $stmt = $db->prepare("INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)");
        $stmt->execute([$username, $email, $hash, $fullName]);
        $userId = (int)$db->lastInsertId();

        // Init settings
        $db->prepare("INSERT INTO user_settings (user_id) VALUES (?)")->execute([$userId]);

        // Init default devices
        $devices = [
            ['lampu-utama', 'Lampu Utama', 'fa-lightbulb', 'iotzy/device/lampu-utama/status', 'iotzy/device/lampu-utama/control'],
            ['kipas-ruangan', 'Kipas Ruangan', 'fa-wind', 'iotzy/device/kipas-ruangan/status', 'iotzy/device/kipas-ruangan/control'],
            ['pintu-utama', 'Pintu Utama', 'fa-lock', 'iotzy/device/pintu-utama/status', 'iotzy/device/pintu-utama/control'],
        ];
        $dStmt = $db->prepare("INSERT INTO devices (user_id, device_key, name, icon, type, topic_sub, topic_pub) VALUES (?,?,?,?,?,?,?)");
        foreach ($devices as $d) {
            $dStmt->execute([$userId, $d[0], $d[1], $d[2], 'switch', $d[3], $d[4]]);
        }

        // Init default sensors
        $sensors = [
            ['temp', 'Suhu Ruangan', 'temperature', 'fa-temperature-half', '°C', 'iotzy/sensor/temperature'],
            ['brightness', 'Kecerahan Ruangan', 'brightness', 'fa-sun', '', 'iotzy/sensor/brightness'],
            ['presence', 'Deteksi Manusia', 'presence', 'fa-user-check', '', 'iotzy/sensor/presence'],
        ];
        $sStmt = $db->prepare("INSERT INTO sensors (user_id, sensor_key, name, type, icon, unit, topic) VALUES (?,?,?,?,?,?,?)");
        foreach ($sensors as $s) {
            $sStmt->execute([$userId, $s[0], $s[1], $s[2], $s[3], $s[4], $s[5]]);
        }

        $db->commit();
    } catch (Throwable $e) {
        if ($db->inTransaction()) {
            $db->rollBack();
        }
        return ['success' => false, 'message' => 'Gagal membuat akun. Silakan coba lagi.'];
    }

    return ['success' => true, 'message' => 'Akun berhasil dibuat! Silakan login.'];
}

// ==================== USER DATA ====================
function getUserDevices(int $userId): array {
    $db = getDB();
    $stmt = $db->prepare("SELECT * FROM devices WHERE user_id = ? ORDER BY created_at ASC");
    $stmt->execute([$userId]);
    return $stmt->fetchAll();
}

function getUserSensors(int $userId): array {
    $db = getDB();
    $stmt = $db->prepare("SELECT * FROM sensors WHERE user_id = ? ORDER BY created_at ASC");
    $stmt->execute([$userId]);
    return $stmt->fetchAll();
}

function getUserSettings(int $userId): array {
    $db = getDB();
    $stmt = $db->prepare("SELECT * FROM user_settings WHERE user_id = ?");
    $stmt->execute([$userId]);
    $row = $stmt->fetch();
    if (!$row) {
        $db->prepare("INSERT INTO user_settings (user_id) VALUES (?)")->execute([$userId]);
        $stmt->execute([$userId]);
        $row = $stmt->fetch();
    }
    return $row;
}

function getUserLogs(int $userId, int $limit = 100): array {
    $db = getDB();
    $limit = max(1, min($limit, 500));
    $stmt = $db->prepare("SELECT * FROM activity_logs WHERE user_id = :uid ORDER BY created_at DESC LIMIT :lim");
    $stmt->bindValue(':uid', $userId, PDO::PARAM_INT);
    $stmt->bindValue(':lim', $limit, PDO::PARAM_INT);
    $stmt->execute();
    return $stmt->fetchAll();
}

function addActivityLog(int $userId, string $device, string $activity, string $trigger, string $type = 'info'): void {
    $db = getDB();
    $db->prepare("INSERT INTO activity_logs (user_id, device_name, activity, trigger_type, log_type) VALUES (?,?,?,?,?)")
       ->execute([$userId, $device, $activity, $trigger, $type]);
}
