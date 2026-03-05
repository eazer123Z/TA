// ==================== AUTOMATION ENGINE MODULE ====================
class AutomationEngine {
    constructor() {
        this.isActive  = false;
        this.rules     = {
            humanDetection: { enabled: true, onDetect: [], onAbsent: [], delay: 3000 },
            lightCondition: { enabled: true, onDark:   [], onBright: [], delay: 2000 },
            schedules:      []
        };

        this.humanTimer         = null;
        this.lightTimer         = null;
        this.lastHuman          = null; // null = unknown, true/false = known state
        this.lastLight          = null;
        this.actionLog          = [];
        this._scheduleInterval  = null;

        // Runtime control untuk aksi agar smooth, tidak spam, dan tetap realtime.
        this._pendingActions    = new Map(); // key: deviceId -> { state, reason, dueAt }
        this._flushTimer        = null;
        this._lastDeviceApplyAt = new Map(); // key: deviceId -> ts
        this._ruleRuntime       = new Map(); // key: sensorId_ruleId -> { active }

        // Jeda minimum antar apply pada device yang sama (anti jitter/spam).
        this.minActionGapMs     = 250;
        this.flushIntervalMs    = 120;
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
        this._ensureFlushLoop();

        console.log('▶️ AutomationEngine started');
    }

    stop() {
        this.isActive = false;

        if (this.humanTimer)        clearTimeout(this.humanTimer);
        if (this.lightTimer)        clearTimeout(this.lightTimer);
        if (this._scheduleInterval) clearInterval(this._scheduleInterval);
        if (this._flushTimer)       clearInterval(this._flushTimer);

        this.humanTimer        = null;
        this.lightTimer        = null;
        this._scheduleInterval = null;
        this._flushTimer       = null;

        this._pendingActions.clear();

        // Reset last-known states so rules can fire again after restart
        this.lastHuman = null;
        this.lastLight = null;
    }

    // ── Human detection handler ──────────────────────────────────
    _onHuman(detected) {
        if (!this.rules.humanDetection.enabled) return;
        if (this.lastHuman === detected) return;

        clearTimeout(this.humanTimer);
        this.humanTimer = setTimeout(() => {
            this.lastHuman = detected;
            const targets = detected
                ? this.rules.humanDetection.onDetect
                : this.rules.humanDetection.onAbsent;
            const reason  = detected ? 'CV: Manusia Terdeteksi' : 'CV: Tidak Ada Manusia';
            this._queueActions(targets, detected, reason);
        }, this._normalizeDelay(this.rules.humanDetection.delay, 3000));
    }

    // ── Light condition handler ──────────────────────────────────
    _onLight(condition) {
        if (!this.rules.lightCondition.enabled) return;
        if (this.lastLight === condition) return;

        clearTimeout(this.lightTimer);
        this.lightTimer = setTimeout(() => {
            this.lastLight = condition;
            if (condition === 'dark') {
                this._queueActions(this.rules.lightCondition.onDark, true, 'CV: Kondisi Gelap');
            } else if (condition === 'bright') {
                this._queueActions(this.rules.lightCondition.onBright, false, 'CV: Kondisi Terang');
            }
            // 'normal' tidak memicu aksi
        }, this._normalizeDelay(this.rules.lightCondition.delay, 2000));
    }

    // ── Sensor value evaluation ──────────────────────────────────
    evaluateSensorRules(sensorId, value) {
        if (!this.isActive) return;

        const sid = String(sensorId);
        const sensorRules = (typeof STATE !== 'undefined' && STATE.automationRules)
            ? (STATE.automationRules[sid] || [])
            : [];

        const numericValue = parseFloat(value);

        sensorRules.forEach(rule => {
            if (!rule || !rule.enabled) return;

            const runtimeKey = `${sid}_${rule.ruleId || rule.deviceId || 'rule'}`;
            const runtime    = this._ruleRuntime.get(runtimeKey) || { active: false };

            let triggered = false;
            switch (rule.condition) {
                case 'gt':
                    triggered = Number.isFinite(numericValue) && numericValue > parseFloat(rule.threshold);
                    break;
                case 'lt':
                    triggered = Number.isFinite(numericValue) && numericValue < parseFloat(rule.threshold);
                    break;
                case 'eq':
                    triggered = Number.isFinite(numericValue) && numericValue === parseFloat(rule.threshold);
                    break;
                case 'range':
                    triggered = Number.isFinite(numericValue) && (
                        numericValue < parseFloat(rule.thresholdMin) ||
                        numericValue > parseFloat(rule.thresholdMax)
                    );
                    break;
                case 'detected':
                    triggered = this._asBoolean(value);
                    break;
                case 'absent':
                    triggered = !this._asBoolean(value);
                    break;
                default:
                    triggered = false;
            }

            // Edge-trigger: hanya fire saat transisi false -> true.
            if (!triggered) {
                runtime.active = false;
                this._ruleRuntime.set(runtimeKey, runtime);
                return;
            }
            if (runtime.active) return;

            // Per-rule cooldown.
            const cooldown = Math.max(this._normalizeDelay(rule.delay, 0) + 2500, 2500);
            const now      = Date.now();
            if (rule._lastFired && (now - rule._lastFired < cooldown)) {
                return;
            }

            runtime.active = true;
            this._ruleRuntime.set(runtimeKey, runtime);
            rule._lastFired = now;

            const state  = rule.action === 'on';
            const sensor = (typeof STATE !== 'undefined') ? STATE.sensors[sid] : null;
            const label  = sensor ? sensor.name : `Sensor ${sid}`;
            const reason = `Sensor Auto(${label})`;

            const delayMs = this._normalizeDelay(rule.delay, 0);
            if (delayMs > 0) {
                setTimeout(() => {
                    if (!this.isActive) return;
                    this._queueActions([rule.deviceId], state, reason);
                }, delayMs);
            } else {
                this._queueActions([rule.deviceId], state, reason);
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
            this._queueActions(s.devices || [], state, `Jadwal: ${s.label || s.time}`);
        });
    }

    // ── Action queue (smooth + realtime) ────────────────────────
    _queueActions(deviceIds, state, reason) {
        if (!this.isActive) return;
        if (!Array.isArray(deviceIds) || !deviceIds.length) return;

        const now = Date.now();
        deviceIds.forEach(rawId => {
            const deviceId = String(rawId);
            if (!deviceId) return;

            const existing = this._pendingActions.get(deviceId);
            const next = {
                state: !!state,
                reason: reason || 'Automation',
                dueAt: now
            };

            // Coalesce: aksi terakhir menggantikan aksi lama agar tidak berkedip.
            if (existing) {
                existing.state = next.state;
                existing.reason = next.reason;
                existing.dueAt = Math.min(existing.dueAt, next.dueAt);
                this._pendingActions.set(deviceId, existing);
            } else {
                this._pendingActions.set(deviceId, next);
            }
        });

        this._ensureFlushLoop();
        this._flushActions();
    }

    _ensureFlushLoop() {
        if (this._flushTimer || !this.isActive) return;
        this._flushTimer = setInterval(() => this._flushActions(), this.flushIntervalMs);
    }

    _flushActions() {
        if (!this.isActive) return;
        if (!this._pendingActions.size) return;
        if (typeof STATE === 'undefined') return;

        const now = Date.now();

        for (const [deviceId, action] of this._pendingActions.entries()) {
            if (!action || now < action.dueAt) continue;

            const device = STATE.devices?.[deviceId];
            if (!device) {
                this._pendingActions.delete(deviceId);
                continue;
            }

            if (STATE.deviceStates?.[deviceId] === action.state) {
                this._pendingActions.delete(deviceId);
                continue;
            }

            const lastApplyAt = this._lastDeviceApplyAt.get(deviceId) || 0;
            if (now - lastApplyAt < this.minActionGapMs) {
                continue;
            }

            this._applyDeviceAction(deviceId, action.state, action.reason);
            this._lastDeviceApplyAt.set(deviceId, now);
            this._pendingActions.delete(deviceId);
        }
    }

    _applyDeviceAction(deviceId, state, reason) {
        const device = STATE.devices?.[deviceId];
        if (!device) return;

        if (typeof applyDeviceState === 'function') {
            applyDeviceState(deviceId, state, reason);
        } else {
            STATE.deviceStates[deviceId] = state;
            if (typeof updateDeviceUI === 'function') updateDeviceUI(deviceId);

            const topics = STATE.deviceTopics?.[deviceId];
            if (topics?.pub && typeof publishMQTT === 'function') {
                publishMQTT(topics.pub, { state: state ? 1 : 0 });
            }
            if (typeof addLog === 'function') {
                addLog(device.name, `Auto ${state ? 'ON' : 'OFF'}`, reason, state ? 'success' : 'info');
            }
        }

        this.actionLog.unshift({ deviceId, name: device.name, state, reason, ts: Date.now() });
        if (this.actionLog.length > 100) this.actionLog.length = 100;

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
        if (s) {
            s.enabled = enabled;
            this._saveToStorage();
        }
    }

    // ── Helpers ──────────────────────────────────────────────────
    _asBoolean(v) {
        if (typeof v === 'boolean') return v;
        if (typeof v === 'number') return v > 0;
        const s = String(v ?? '').trim().toLowerCase();
        if (!s) return false;
        return ['1', 'true', 'on', 'yes', 'detected'].includes(s);
    }

    _normalizeDelay(delay, fallback) {
        const n = Number(delay);
        if (!Number.isFinite(n) || n < 0) return fallback;
        return Math.floor(n);
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

    getActionLog() {
        return this.actionLog.slice(0, 50);
    }

    destroy() {
        this.stop();
        this.actionLog = [];
        this._ruleRuntime.clear();
        this._lastDeviceApplyAt.clear();
    }
}

const automationEngine = new AutomationEngine();
