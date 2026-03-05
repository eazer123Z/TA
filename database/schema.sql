CREATE DATABASE IF NOT EXISTS iotzy_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE iotzy_db;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  email VARCHAR(120) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  full_name VARCHAR(120) DEFAULT '',
  role ENUM('admin','user') DEFAULT 'user',
  is_active TINYINT(1) DEFAULT 1,
  last_login DATETIME NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_settings (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL UNIQUE,
  mqtt_broker VARCHAR(255) DEFAULT 'broker.hivemq.com',
  mqtt_port INT DEFAULT 8884,
  mqtt_client_id VARCHAR(120) DEFAULT 'iotzy_web',
  mqtt_path VARCHAR(120) DEFAULT '/mqtt',
  mqtt_use_ssl TINYINT(1) DEFAULT 1,
  mqtt_username VARCHAR(120) DEFAULT '',
  quick_control_devices JSON NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS devices (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  name VARCHAR(120) NOT NULL,
  icon VARCHAR(64) DEFAULT 'fa-plug',
  type VARCHAR(40) DEFAULT 'switch',
  topic_sub VARCHAR(255) DEFAULT '',
  topic_pub VARCHAR(255) DEFAULT '',
  device_key VARCHAR(64) UNIQUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sensors (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  name VARCHAR(120) NOT NULL,
  type VARCHAR(50) DEFAULT 'temperature',
  icon VARCHAR(64) DEFAULT 'fa-microchip',
  topic VARCHAR(255) NOT NULL,
  unit VARCHAR(32) DEFAULT '',
  sensor_key VARCHAR(64) UNIQUE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS activity_logs (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  device_name VARCHAR(120) DEFAULT 'System',
  activity TEXT NOT NULL,
  trigger_type VARCHAR(120) DEFAULT 'System',
  log_type VARCHAR(20) DEFAULT 'info',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_user_created (user_id, created_at),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
