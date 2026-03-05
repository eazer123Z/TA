<?php
require_once __DIR__ . '/includes/auth.php';
startSecureSession();
if (isLoggedIn()) { header('Location: index.php'); exit; }
$error=''; $success=''; $mode=$_GET['mode'] ?? 'login';
if ($_SERVER['REQUEST_METHOD']==='POST') {
  $action=$_POST['action'] ?? 'login';
  if ($action==='login') {
    $r=loginUser(trim($_POST['username']??''), $_POST['password']??'');
    if ($r['success']) { header('Location: index.php'); exit; }
    $error=$r['message'];
  } else {
    if (($_POST['password']??'') !== ($_POST['password2']??'')) { $error='Konfirmasi password tidak cocok.'; $mode='register'; }
    else {
      $r=registerUser(trim($_POST['username']??''), trim($_POST['email']??''), $_POST['password']??'', trim($_POST['full_name']??''));
      if ($r['success']) { $success=$r['message']; $mode='login'; } else { $error=$r['message']; $mode='register'; }
    }
  }
}
?>
<!doctype html><html lang="id"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IoTzy Login</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"><style>body{font-family:Arial;background:#f6f7fb;display:grid;place-items:center;min-height:100vh}.box{background:#fff;border:1px solid #ddd;border-radius:12px;padding:24px;max-width:420px;width:100%}input{width:100%;padding:10px;margin:8px 0}button{padding:10px 14px;background:#111;color:#fff;border:0;border-radius:8px}.err{color:#b91c1c}.ok{color:#15803d}</style></head><body>
<div class="box"><h2><?= $mode==='register'?'Daftar':'Masuk' ?></h2><?php if($error):?><p class="err"><?= htmlspecialchars($error) ?></p><?php endif; ?><?php if($success):?><p class="ok"><?= htmlspecialchars($success) ?></p><?php endif; ?>
<?php if($mode==='login'): ?>
<form method="post"><input type="hidden" name="action" value="login"><input name="username" placeholder="username/email" required><input type="password" name="password" placeholder="password" required><button>Masuk</button></form><p>Belum punya akun? <a href="?mode=register">Daftar</a></p>
<?php else: ?>
<form method="post"><input type="hidden" name="action" value="register"><input name="full_name" placeholder="nama lengkap"><input name="username" placeholder="username" required><input name="email" type="email" placeholder="email" required><input type="password" name="password" placeholder="password" required><input type="password" name="password2" placeholder="konfirmasi password" required><button>Daftar</button></form><p>Sudah punya akun? <a href="?mode=login">Masuk</a></p>
<?php endif; ?></div></body></html>
