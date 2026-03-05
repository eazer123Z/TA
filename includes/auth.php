<?php
require_once __DIR__ . '/../config.php';

function startSecureSession(): void {
    if (session_status() === PHP_SESSION_NONE) {
        session_set_cookie_params([
            'lifetime' => SESSION_LIFETIME,
            'path' => '/',
            'secure' => false,
            'httponly' => true,
            'samesite' => 'Strict',
        ]);
        session_start();
    }
}

function isLoggedIn(): bool {
    startSecureSession();
    return isset($_SESSION['user_id'], $_SESSION['token']);
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
    $s = $db->prepare('SELECT id, username, email, full_name, role FROM users WHERE id=? AND is_active=1');
    $s->execute([$_SESSION['user_id']]);
    return $s->fetch() ?: null;
}

function loginUser(string $username, string $password): array {
    $db = getDB();
    $s = $db->prepare('SELECT * FROM users WHERE (username=? OR email=?) AND is_active=1 LIMIT 1');
    $s->execute([$username, $username]);
    $u = $s->fetch();
    if (!$u || !password_verify($password, $u['password_hash'])) return ['success'=>false,'message'=>'Username atau password salah.'];
    startSecureSession();
    session_regenerate_id(true);
    $_SESSION['user_id'] = $u['id'];
    $_SESSION['username'] = $u['username'];
    $_SESSION['role'] = $u['role'];
    $_SESSION['token'] = bin2hex(random_bytes(32));
    $db->prepare('UPDATE users SET last_login=NOW() WHERE id=?')->execute([$u['id']]);
    $db->prepare('INSERT IGNORE INTO user_settings (user_id) VALUES (?)')->execute([$u['id']]);
    return ['success'=>true,'message'=>'Login berhasil!'];
}

function logoutUser(): void {
    startSecureSession();
    $_SESSION = [];
    session_destroy();
}

function registerUser(string $username, string $email, string $password, string $fullName): array {
    if (!preg_match('/^[a-zA-Z0-9_]{3,50}$/', $username)) return ['success'=>false,'message'=>'Username tidak valid.'];
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) return ['success'=>false,'message'=>'Email tidak valid.'];
    if (strlen($password) < 8) return ['success'=>false,'message'=>'Password minimal 8 karakter.'];
    $db = getDB();
    $s = $db->prepare('SELECT id FROM users WHERE username=? OR email=?');
    $s->execute([$username, $email]);
    if ($s->fetch()) return ['success'=>false,'message'=>'Username/email sudah digunakan.'];
    $h = password_hash($password, PASSWORD_BCRYPT, ['cost'=>12]);
    $db->prepare('INSERT INTO users (username,email,password_hash,full_name) VALUES (?,?,?,?)')->execute([$username,$email,$h,$fullName]);
    $uid = (int)$db->lastInsertId();
    $db->prepare('INSERT INTO user_settings (user_id) VALUES (?)')->execute([$uid]);
    return ['success'=>true,'message'=>'Akun berhasil dibuat!'];
}

function getUserDevices(int $userId): array {
    $s = getDB()->prepare('SELECT * FROM devices WHERE user_id=? ORDER BY created_at ASC');
    $s->execute([$userId]); return $s->fetchAll();
}
function getUserSensors(int $userId): array {
    $s = getDB()->prepare('SELECT * FROM sensors WHERE user_id=? ORDER BY created_at ASC');
    $s->execute([$userId]); return $s->fetchAll();
}
function getUserSettings(int $userId): array {
    $db = getDB();
    $s = $db->prepare('SELECT * FROM user_settings WHERE user_id=?');
    $s->execute([$userId]);
    $r = $s->fetch();
    if (!$r) { $db->prepare('INSERT INTO user_settings (user_id) VALUES (?)')->execute([$userId]); $s->execute([$userId]); $r = $s->fetch(); }
    return $r;
}
function addActivityLog(int $userId, string $device, string $activity, string $trigger, string $type='info'): void {
    getDB()->prepare('INSERT INTO activity_logs (user_id,device_name,activity,trigger_type,log_type) VALUES (?,?,?,?,?)')
        ->execute([$userId,$device,$activity,$trigger,$type]);
}
