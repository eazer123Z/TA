// ==================== AUTOMATION ENGINE MODULE ====================
class AutomationEngine {
    constructor() {
        this.isActive  = false;
        this.rules     = {
            humanDetection: { enabled: true, onDetect: [], onAbsent: [], delay: 3000 },
            lightCondition: { enabled: true, onDark:   [], onBright: [], delay: 2000 },
            schedules:      []
        };
        this.humanTimer        = null;
        this.lightTimer        = null;
        this.lastHuman         = null;  // null = unknown, true/false = known state
        this.lastLight         = null;
        this.actionLog         = [];
        this._scheduleInterval = null;
    }

    // ── Initialize ───────────────────────────────────────────────
    initialize() {
        this._loadFromStorage();
        console.log('⚙️ AutomationEngine ready');
    }

    // ── Start/Stop ───────────────────────────────────────────────
    start() {
        if (this.isActive) return;
        this.isActive = true;

        // Register callbacks with unique tag so they never overwrite cvUI callbacks
        if (typeof cvDetector !== 'undefined') {
            cvDetector.setCallbacks({
                _tag:            'automationEngine',
                onHumanDetected: () => this._onHuman(true),
                onHumanAbsent:   () => this._onHuman(false)
            });
        }
        if (typeof lightAnalyzer !== 'undefined') {
            lightAnalyzer.setCallbacks({
                _tag:          'automationEngine',
                onLightChange: (cond) => this._onLight(cond)
            });
        }

        // Schedule checker every 30 s
        this._scheduleInterval = setInterval(() => this._checkSchedules(), 30000);
        this._checkSchedules();
        console.log('▶️ AutomationEngine started');
    }

    stop() {
        this.isActive = false;
        if (this.humanTimer)        clearTimeout(this.humanTimer);
        if (this.lightTimer)        clearTimeout(this.lightTimer);
        if (this._scheduleInterval) clearInterval(this._scheduleInterval);
        this.humanTimer        = null;
        this.lightTimer        = null;
        this._scheduleInterval = null;
        // Reset last-known states so rules can fire again after restart
        this.lastHuman = null;
        this.lastLight = null;
    }

    // ── Human detection handler ──────────────────────────────────
    _onHuman(detected) {
        if (!this.rules.humanDetection.enabled) return;
        if (this.lastHuman === detected) return; // no state change
        clearTimeout(this.humanTimer);
        this.humanTimer = setTimeout(() => {
            this.lastHuman = detected;
            const targets = detected
                ? this.rules.humanDetection.onDetect
                : this.rules.humanDetection.onAbsent;
            const reason  = detected ? 'CV: Manusia Terdeteksi' : 'CV: Tidak Ada Manusia';
            this._execute(targets, detected, reason);
        }, this.rules.humanDetection.delay || 3000);
    }

    // ── Light condition handler ──────────────────────────────────
    _onLight(condition) {
        if (!this.rules.lightCondition.enabled) return;
        if (this.lastLight === condition) return; // no change
        clearTimeout(this.lightTimer);
        this.lightTimer = setTimeout(() => {
            this.lastLight = condition;
            if (condition === 'dark')
                this._execute(this.rules.lightCondition.onDark,   true,  'CV: Kondisi Gelap');
            else if (condition === 'bright')
                this._execute(this.rules.lightCondition.onBright, false, 'CV: Kondisi Terang');
            // 'normal' condition intentionally does nothing
        }, this.rules.lightCondition.delay || 2000);
    }

    // ── Sensor value evaluation ──────────────────────────────────
    evaluateSensorRules(sensorId, value) {
        if (!this.isActive) return;
        const sensorRules = (typeof STATE !== 'undefined' && STATE.automationRules)
            ? (STATE.automationRules[String(sensorId)] || [])
            : [];

        sensorRules.forEach(rule => {
            if (!rule.enabled) return;
            let triggered = false;
            const v       = parseFloat(value);
            switch (rule.condition) {
                case 'gt':       triggered = v >  parseFloat(rule.threshold); break;
                case 'lt':       triggered = v <  parseFloat(rule.threshold); break;
                case 'eq':       triggered = v === parseFloat(rule.threshold); break;
                case 'range':    triggered = v <  parseFloat(rule.thresholdMin) || v > parseFloat(rule.thresholdMax); break;
                case 'detected': triggered = !!value;  break;
                case 'absent':   triggered = !value;   break;
                default:         break;
            }
            if (!triggered) return;

            // Per-rule cooldown: min of rule.delay + 3 s, at least 3 s
            const cooldown = Math.max((rule.delay || 0) + 3000, 3000);
            const now      = Date.now();
            if (rule._lastFired && now - rule._lastFired < cooldown) return;
            rule._lastFired = now;

            const state  = rule.action === 'on';
            const sensor = (typeof STATE !== 'undefined') ? STATE.sensors[String(sensorId)] : null;
            const label  = sensor ? sensor.name : `Sensor ${sensorId}`;
            const reason = `Sensor Auto(${label})`;

            if (rule.delay > 0) {
                setTimeout(() => this._execute([rule.deviceId], state, reason), rule.delay);
            } else {
                this._execute([rule.deviceId], state, reason);
            }
        });
    }

    // ── Schedule checker ─────────────────────────────────────────
    _checkSchedules() {
        if (!this.isActive) return;
        const schedules = this.rules.schedules || [];
        if (!schedules.length) return;

        const now  = new Date();
        const hhmm = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
        const dow  = now.getDay();

        schedules.forEach(s => {
            if (!s.enabled) return;
            if (s.time !== hhmm) return;
            if (s.days && s.days.length && !s.days.includes(dow)) return;

            // Fire only once per minute per schedule using sessionStorage
            const key = `sched_${s.id}_${hhmm}_${now.toDateString()}`;
            try {
                if (sessionStorage.getItem(key)) return;
                sessionStorage.setItem(key, '1');
            } catch (_) {}

            const state = s.action !== 'off';
            this._execute(s.devices || [], state, `Jadwal: ${s.label || s.time}`);
        });
    }

    // ── Execute device actions ───────────────────────────────────
    _execute(deviceIds, state, reason) {
        if (!deviceIds || !deviceIds.length) return;
        if (typeof STATE === 'undefined') return;

        deviceIds.forEach(rawId => {
            const deviceId = String(rawId);
            const device   = STATE.devices[deviceId];
            if (!device) return;
            if (STATE.deviceStates[deviceId] === state) return;

            if (typeof applyDeviceState === 'function') {
                applyDeviceState(deviceId, state, reason);
            } else {
                STATE.deviceStates[deviceId] = state;
                if (typeof updateDeviceUI === 'function')   updateDeviceUI(deviceId);
                const topics = STATE.deviceTopics?.[deviceId];
                if (topics?.pub && typeof publishMQTT === 'function')
                    publishMQTT(topics.pub, { state: state ? 1 : 0 });
                if (typeof addLog === 'function')
                    addLog(device.name, `Auto ${state ? 'ON' : 'OFF'}`, reason, state ? 'success' : 'info');
            }

            this.actionLog.unshift({ deviceId, name: device.name, state, reason, ts: Date.now() });
            if (this.actionLog.length > 100) this.actionLog.length = 100;
        });

        if (typeof updateDashboardStats === 'function') updateDashboardStats();
    }

    // ── CV rule management ───────────────────────────────────────
    updateCVRules(patch) {
        if (patch.humanDetection) Object.assign(this.rules.humanDetection, patch.humanDetection);
        if (patch.lightCondition) Object.assign(this.rules.lightCondition,  patch.lightCondition);
        this._saveToStorage();
        // Re-register callbacks so delays/lists are up-to-date
        if (this.isActive) {
            this.stop();
            this.start();
        }
    }

    getCVRules() {
        return JSON.parse(JSON.stringify({
            humanDetection: this.rules.humanDetection,
            lightCondition: this.rules.lightCondition
        }));
    }

    setEnabled(type, enabled) {
        if (this.rules[type] !== undefined) {
            this.rules[type].enabled = enabled;
            this._saveToStorage();
        }
    }

    // ── Schedule management ──────────────────────────────────────
    getSchedules() {
        return JSON.parse(JSON.stringify(this.rules.schedules || []));
    }

    addSchedule(time, days, devices, action, label) {
        if (!this.rules.schedules) this.rules.schedules = [];
        const s = {
            id:      `s_${Date.now()}`,
            enabled: true,
            time, days, devices, action,
            label:   label || `Jadwal ${time}`
        };
        this.rules.schedules.push(s);
        this._saveToStorage();
        return s;
    }

    removeSchedule(id) {
        if (!this.rules.schedules) return;
        this.rules.schedules = this.rules.schedules.filter(s => s.id !== id);
        this._saveToStorage();
    }

    toggleSchedule(id, enabled) {
        const s = (this.rules.schedules || []).find(x => x.id === id);
        if (s) { s.enabled = enabled; this._saveToStorage(); }
    }

    // ── Persistence ──────────────────────────────────────────────
    _saveToStorage() {
        try {
            localStorage.setItem('iotzy_auto_engine_v2', JSON.stringify({
                humanDetection: this.rules.humanDetection,
                lightCondition: this.rules.lightCondition,
                schedules:      this.rules.schedules
            }));
        } catch (_) {}
    }

    _loadFromStorage() {
        try {
            const raw = localStorage.getItem('iotzy_auto_engine_v2');
            if (!raw) return;
            const saved = JSON.parse(raw);
            if (saved.humanDetection) Object.assign(this.rules.humanDetection, saved.humanDetection);
            if (saved.lightCondition) Object.assign(this.rules.lightCondition,  saved.lightCondition);
            if (Array.isArray(saved.schedules)) this.rules.schedules = saved.schedules;
        } catch (_) {}
    }

    getActionLog() { return this.actionLog.slice(0, 50); }

    destroy() {
        this.stop();
        this.actionLog = [];
    }
}

const automationEngine = new AutomationEngine();
