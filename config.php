<?php
// ==================== DATABASE CONFIGURATION ====================
// Copy this file to config.php and fill in your details

define('DB_HOST', 'localhost');
define('DB_NAME', 'iotzy_db');
define('DB_USER', 'root');          // Change to your DB username
define('DB_PASS', '');              // Change to your DB password
define('DB_CHARSET', 'utf8mb4');

define('APP_NAME', 'IoTzy');
define('APP_URL', 'http://localhost/iotzy');  // Change to your URL
define('APP_SECRET', 'change_this_to_random_string_32chars!!');  // Change this!
define('APP_ENV', getenv('APP_ENV') ?: 'development');
define('SESSION_LIFETIME', 86400);  // 24 hours

function isProduction(): bool {
    return APP_ENV === 'production';
}

// PDO Connection
function getDB(): PDO {
    static $pdo = null;
    if ($pdo === null) {
        $dsn = "mysql:host=" . DB_HOST . ";dbname=" . DB_NAME . ";charset=" . DB_CHARSET;
        $options = [
            PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES   => false,
        ];
        try {
            $pdo = new PDO($dsn, DB_USER, DB_PASS, $options);
        } catch (PDOException $e) {
            http_response_code(500);
            header('Content-Type: application/json; charset=utf-8');
            $message = isProduction()
                ? 'Database connection failed'
                : ('Database connection failed: ' . $e->getMessage());
            echo json_encode(['success' => false, 'error' => $message]);
            exit;
        }
    }
    return $pdo;
}
