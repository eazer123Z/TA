<?php
require_once __DIR__ . '/../includes/auth.php';
header('Content-Type: application/json; charset=utf-8');

if (!isLoggedIn()) {
    http_response_code(401);
    echo json_encode(['success' => false, 'error' => 'Unauthorized']);
    exit;
}

$user = getCurrentUser();
if (!$user) {
    http_response_code(401);
    echo json_encode(['success' => false, 'error' => 'Unauthorized']);
    exit;
}

$uid = (int)$user['id'];
$action = (string)($_GET['action'] ?? '');
$input = json_decode(file_get_contents('php://input'), true);
if (!is_array($input)) {
    $input = [];
}
$db = getDB();

$mutatingActions = [
    'add_log', 'clear_logs',
    'add_device', 'update_device', 'delete_device',
    'add_sensor', 'update_sensor', 'delete_sensor',
    'save_settings', 'update_profile', 'change_password',
];

try {
    if (in_array($action, $mutatingActions, true) && $_SERVER['REQUEST_METHOD'] !== 'POST') {
        throw new RuntimeException('Method not allowed');
    }

    switch ($action) {
        case 'get_logs': {
            $s = $db->prepare('SELECT *, DATE(created_at) AS tanggal, TIME(created_at) AS waktu FROM activity_logs WHERE user_id=? ORDER BY id DESC LIMIT 500');
            $s->execute([$uid]);
            echo json_encode($s->fetchAll());
            break;
        }
        case 'add_log': {
            addActivityLog(
                $uid,
                (string)($input['device'] ?? 'System'),
                (string)($input['activity'] ?? '-'),
                (string)($input['trigger'] ?? 'System'),
                (string)($input['type'] ?? 'info')
            );
            echo json_encode(['success' => true]);
            break;
        }
        case 'clear_logs': {
            $db->prepare('DELETE FROM activity_logs WHERE user_id=?')->execute([$uid]);
            echo json_encode(['success' => true]);
            break;
        }

        case 'add_device': {
            $name = trim((string)($input['name'] ?? ''));
            if ($name === '') throw new RuntimeException('Name required');
            $icon = (string)($input['icon'] ?? 'fa-plug');
            $sub = (string)($input['topic_sub'] ?? '');
            $pub = (string)($input['topic_pub'] ?? '');
            $key = bin2hex(random_bytes(12));
            $s = $db->prepare('INSERT INTO devices(user_id,name,icon,type,topic_sub,topic_pub,device_key) VALUES(?,?,?,?,?,?,?)');
            $s->execute([$uid, $name, $icon, 'switch', $sub, $pub, $key]);
            echo json_encode(['success' => true, 'id' => $db->lastInsertId(), 'device_key' => $key]);
            break;
        }
        case 'update_device': {
            $id = (int)($input['id'] ?? 0);
            $name = trim((string)($input['name'] ?? ''));
            if ($id <= 0 || $name === '') throw new RuntimeException('Invalid device');
            $icon = (string)($input['icon'] ?? 'fa-plug');
            $sub = (string)($input['topic_sub'] ?? '');
            $pub = (string)($input['topic_pub'] ?? '');
            $s = $db->prepare('UPDATE devices SET name=?, icon=?, topic_sub=?, topic_pub=? WHERE id=? AND user_id=?');
            $s->execute([$name, $icon, $sub, $pub, $id, $uid]);
            echo json_encode(['success' => true]);
            break;
        }
        case 'delete_device': {
            $id = (int)($input['id'] ?? 0);
            if ($id <= 0) throw new RuntimeException('Invalid device');
            $s = $db->prepare('DELETE FROM devices WHERE id=? AND user_id=?');
            $s->execute([$id, $uid]);
            echo json_encode(['success' => true]);
            break;
        }

        case 'add_sensor': {
            $name = trim((string)($input['name'] ?? ''));
            $topic = trim((string)($input['topic'] ?? ''));
            if ($name === '' || $topic === '') throw new RuntimeException('Invalid input');
            $type = (string)($input['type'] ?? 'temperature');
            $unit = (string)($input['unit'] ?? '');
            $key = bin2hex(random_bytes(12));
            $s = $db->prepare('INSERT INTO sensors(user_id,name,type,topic,unit,sensor_key,icon) VALUES(?,?,?,?,?,?,?)');
            $s->execute([$uid, $name, $type, $topic, $unit, $key, 'fa-microchip']);
            echo json_encode(['success' => true, 'id' => $db->lastInsertId(), 'sensor_key' => $key]);
            break;
        }
        case 'update_sensor': {
            $id = (int)($input['id'] ?? 0);
            $name = trim((string)($input['name'] ?? ''));
            $topic = trim((string)($input['topic'] ?? ''));
            $type = (string)($input['type'] ?? 'temperature');
            $unit = (string)($input['unit'] ?? '');
            if ($id <= 0 || $name === '' || $topic === '') throw new RuntimeException('Invalid sensor');
            $s = $db->prepare('UPDATE sensors SET name=?, type=?, topic=?, unit=? WHERE id=? AND user_id=?');
            $s->execute([$name, $type, $topic, $unit, $id, $uid]);
            echo json_encode(['success' => true]);
            break;
        }
        case 'delete_sensor': {
            $id = (int)($input['id'] ?? 0);
            if ($id <= 0) throw new RuntimeException('Invalid sensor');
            $s = $db->prepare('DELETE FROM sensors WHERE id=? AND user_id=?');
            $s->execute([$id, $uid]);
            echo json_encode(['success' => true]);
            break;
        }

        case 'save_settings': {
            $broker = (string)($input['mqtt_broker'] ?? 'broker.hivemq.com');
            $port = (int)($input['mqtt_port'] ?? 8884);
            if ($broker === '' || strlen($broker) > 255) {
                throw new RuntimeException('MQTT broker tidak valid');
            }
            if ($port < 1 || $port > 65535) {
                throw new RuntimeException('MQTT port tidak valid');
            }
            $s = $db->prepare('UPDATE user_settings SET mqtt_broker=?, mqtt_port=? WHERE user_id=?');
            $s->execute([$broker, $port, $uid]);
            echo json_encode(['success' => true]);
            break;
        }
        case 'update_profile': {
            $fullName = trim((string)($input['full_name'] ?? ''));
            $email = strtolower(trim((string)($input['email'] ?? '')));
            if ($fullName === '' || !filter_var($email, FILTER_VALIDATE_EMAIL)) throw new RuntimeException('Invalid profile');
            $s = $db->prepare('UPDATE users SET full_name=?, email=? WHERE id=?');
            $s->execute([$fullName, $email, $uid]);
            echo json_encode(['success' => true]);
            break;
        }
        case 'change_password': {
            $current = (string)($input['current_password'] ?? '');
            $next = (string)($input['new_password'] ?? '');
            if (strlen($next) < 8) throw new RuntimeException('Password minimal 8 karakter');
            $s = $db->prepare('SELECT password_hash FROM users WHERE id=? LIMIT 1');
            $s->execute([$uid]);
            $u = $s->fetch();
            if (!$u || !password_verify($current, $u['password_hash'])) throw new RuntimeException('Password saat ini salah');
            $h = password_hash($next, PASSWORD_BCRYPT, ['cost' => 12]);
            $db->prepare('UPDATE users SET password_hash=? WHERE id=?')->execute([$h, $uid]);
            echo json_encode(['success' => true]);
            break;
        }

        default:
            echo json_encode(['success' => false, 'error' => 'Unknown action']);
    }
} catch (Throwable $e) {
    if (($e->getMessage() ?? '') === 'Method not allowed') {
        http_response_code(405);
    } else {
        http_response_code(400);
    }
    echo json_encode(['success' => false, 'error' => $e->getMessage()]);
}
