-- IoTzy Dashboard Database Schema
-- Run this on your MySQL/MariaDB server

CREATE DATABASE IF NOT EXISTS iotzy_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE iotzy_db;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100) DEFAULT NULL,
    role ENUM('admin','user') DEFAULT 'user',
    is_active TINYINT(1) DEFAULT 1,
    last_login DATETIME DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    session_token VARCHAR(128) NOT NULL UNIQUE,
    ip_address VARCHAR(45) DEFAULT NULL,
    user_agent TEXT DEFAULT NULL,
    expires_at DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Devices table
CREATE TABLE IF NOT EXISTS devices (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    device_key VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    icon VARCHAR(50) DEFAULT 'fa-plug',
    type VARCHAR(50) DEFAULT 'switch',
    topic_sub VARCHAR(200) DEFAULT NULL,
    topic_pub VARCHAR(200) DEFAULT NULL,
    is_active TINYINT(1) DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Sensors table
CREATE TABLE IF NOT EXISTS sensors (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    sensor_key VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL,
    icon VARCHAR(50) DEFAULT 'fa-microchip',
    unit VARCHAR(20) DEFAULT NULL,
    topic VARCHAR(200) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Activity logs table
CREATE TABLE IF NOT EXISTS activity_logs (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL,
    device_name VARCHAR(100) NOT NULL,
    activity VARCHAR(200) NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    log_type ENUM('info','success','warning','error') DEFAULT 'info',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- User settings table
CREATE TABLE IF NOT EXISTS user_settings (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNSIGNED NOT NULL UNIQUE,
    mqtt_broker VARCHAR(200) DEFAULT 'broker.hivemq.com',
    mqtt_port INT DEFAULT 8884,
    mqtt_client_id VARCHAR(100) DEFAULT NULL,
    mqtt_path VARCHAR(100) DEFAULT '/mqtt',
    mqtt_use_ssl TINYINT(1) DEFAULT 1,
    mqtt_username VARCHAR(100) DEFAULT NULL,
    mqtt_password_enc VARCHAR(255) DEFAULT NULL,
    automation_lamp TINYINT(1) DEFAULT 1,
    automation_fan TINYINT(1) DEFAULT 1,
    automation_lock TINYINT(1) DEFAULT 1,
    lamp_on_threshold DECIMAL(4,2) DEFAULT 0.35,
    lamp_off_threshold DECIMAL(4,2) DEFAULT 0.50,
    fan_temp_high DECIMAL(5,2) DEFAULT 26.50,
    fan_temp_normal DECIMAL(5,2) DEFAULT 25.00,
    lock_delay INT DEFAULT 5000,
    quick_control_devices JSON DEFAULT NULL,
    theme VARCHAR(10) DEFAULT 'light',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Default admin user (password: Admin@123)
INSERT INTO users (username, email, password_hash, full_name, role) VALUES
('admin', 'admin@iotzy.local', '$2y$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMZJool3xvXBB73QRD8C5BPxjC', 'Administrator', 'admin');

-- Default user (password: User@123)
INSERT INTO users (username, email, password_hash, full_name, role) VALUES
('demo', 'demo@iotzy.local', '$2y$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC7ok9Z.Yt/MvIGW7Ld2', 'Demo User', 'user');
