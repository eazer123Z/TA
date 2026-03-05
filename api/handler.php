<?php
require_once __DIR__ . '/../includes/auth.php';
header('Content-Type: application/json; charset=utf-8');
if (!isLoggedIn()) { http_response_code(401); echo json_encode(['success'=>false,'error'=>'Unauthorized']); exit; }
$user = getCurrentUser();
$uid = (int)$user['id'];
$action = $_GET['action'] ?? '';
$input = json_decode(file_get_contents('php://input'), true) ?: [];
$db = getDB();

try {
  switch ($action) {
    case 'get_logs':
      $s=$db->prepare('SELECT *, DATE(created_at) tanggal, TIME(created_at) waktu FROM activity_logs WHERE user_id=? ORDER BY id DESC LIMIT 500');
      $s->execute([$uid]); echo json_encode($s->fetchAll()); break;
    case 'add_log':
      addActivityLog($uid, (string)($input['device'] ?? 'System'), (string)($input['activity'] ?? '-'), (string)($input['trigger'] ?? 'System'), (string)($input['type'] ?? 'info'));
      echo json_encode(['success'=>true]); break;
    case 'add_device':
      $name=trim((string)($input['name'] ?? '')); if($name==='') throw new RuntimeException('Name required');
      $icon=(string)($input['icon'] ?? 'fa-plug'); $sub=(string)($input['topic_sub'] ?? ''); $pub=(string)($input['topic_pub'] ?? '');
      $key=bin2hex(random_bytes(12));
      $s=$db->prepare('INSERT INTO devices(user_id,name,icon,type,topic_sub,topic_pub,device_key) VALUES(?,?,?,?,?,?,?)');
      $s->execute([$uid,$name,$icon,'switch',$sub,$pub,$key]);
      echo json_encode(['success'=>true,'id'=>$db->lastInsertId(),'device_key'=>$key]); break;
    case 'add_sensor':
      $name=trim((string)($input['name'] ?? '')); $topic=trim((string)($input['topic'] ?? '')); if($name===''||$topic==='') throw new RuntimeException('Invalid input');
      $type=(string)($input['type'] ?? 'temperature'); $unit=(string)($input['unit'] ?? ''); $key=bin2hex(random_bytes(12));
      $s=$db->prepare('INSERT INTO sensors(user_id,name,type,topic,unit,sensor_key,icon) VALUES(?,?,?,?,?,?,?)');
      $s->execute([$uid,$name,$type,$topic,$unit,$key,'fa-microchip']);
      echo json_encode(['success'=>true,'id'=>$db->lastInsertId(),'sensor_key'=>$key]); break;
    case 'save_settings':
      $broker=(string)($input['mqtt_broker'] ?? 'broker.hivemq.com'); $port=(int)($input['mqtt_port'] ?? 8884);
      $s=$db->prepare('UPDATE user_settings SET mqtt_broker=?, mqtt_port=? WHERE user_id=?');
      $s->execute([$broker,$port,$uid]); echo json_encode(['success'=>true]); break;
    default:
      echo json_encode(['success'=>false,'error'=>'Unknown action']);
  }
} catch (Throwable $e) {
  http_response_code(400);
  echo json_encode(['success'=>false,'error'=>$e->getMessage()]);
}
