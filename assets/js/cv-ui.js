// ==================== CV UI MODULE ====================
class CVUI {
    constructor() {
        this.overlayCanvas  = null;
        this.overlayContext = null;
        this._initialized   = false;
    }

    initialize() {
        if (this._initialized) return;
        this._initialized = true;
        this._createOverlay();
        this._hookCallbacks();
        console.log('🎨 CVUI initialized');
    }

    _createOverlay() {
        let c = document.getElementById('cvOverlayCanvas');
        if (!c) {
            c    = document.createElement('canvas');
            c.id = 'cvOverlayCanvas';
        }
        c.style.cssText     = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:10';
        this.overlayCanvas  = c;
        this.overlayContext = c.getContext('2d');
    }

    _hookCallbacks() {
        // Register with unique tag so automation-engine callbacks are NOT overwritten
        if (typeof cvDetector !== 'undefined') {
            cvDetector.setCallbacks({
                _tag:             'cvUI',
                onDetectionUpdate: (d) => this._onDetectionUpdate(d)
            });
        }
        if (typeof lightAnalyzer !== 'undefined') {
            lightAnalyzer.setCallbacks({
                _tag:              'cvUI',
                onBrightnessUpdate: (b, c) => this._onBrightnessUpdate(b, c)
            });
        }
    }

    // ── Detection display ────────────────────────────────────────
    _onDetectionUpdate(data) {
        const hc = document.getElementById('cvHumanCount');
        const ps = document.getElementById('cvPresenceStatus');
        const cf = document.getElementById('cvConfidence');
        if (hc) hc.textContent = data.humanCount;
        if (ps) {
            ps.textContent = data.humanPresent ? 'Terdeteksi' : 'Tidak Terdeteksi';
            ps.className   = 'status-val' + (data.humanPresent ? ' ok' : ' muted');
        }
        if (cf && data.avgConfidence > 0)
            cf.textContent = (data.avgConfidence * 100).toFixed(0) + '%';
        if (CV_CONFIG.ui.showBoundingBoxes)
            this._drawBoxes(data.detections);
    }

    _drawBoxes(detections) {
        if (!this.overlayCanvas || !this.overlayContext) return;
        const video = document.getElementById('cameraFocus') || document.getElementById('camera');
        if (!video) return;
        const c     = this.overlayCanvas;
        c.width     = video.videoWidth  || video.offsetWidth  || 640;
        c.height    = video.videoHeight || video.offsetHeight || 480;
        const ctx   = this.overlayContext;
        const color = CV_CONFIG.ui.overlayColor || '#6366f1';
        ctx.clearRect(0, 0, c.width, c.height);
        detections.forEach(d => {
            const [x, y, w, h] = d.bbox;
            ctx.strokeStyle = color;
            ctx.lineWidth   = 2;
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = color + '14';
            ctx.fillRect(x, y, w, h);
            const label = `${d.class} ${(d.score * 100).toFixed(0)}%`;
            ctx.font     = 'bold 11px "Plus Jakarta Sans", sans-serif';
            const tw     = ctx.measureText(label).width;
            ctx.fillStyle = color + 'dd';
            ctx.fillRect(x, y > 20 ? y - 20 : y, tw + 10, 18);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, x + 5, y > 20 ? y - 5 : y + 12);
        });
    }

    // ── Brightness display ───────────────────────────────────────
    _onBrightnessUpdate(brightness, condition) {
        const pct  = (brightness * 100).toFixed(1) + '%';
        const map  = { dark: '🌙 Gelap', normal: '☁️ Normal', bright: '☀️ Terang' };
        const bEl  = document.getElementById('cvBrightness');
        const blEl = document.getElementById('cvBrightnessLabel');
        const cEl  = document.getElementById('cvLightCondition');
        const bBar = document.getElementById('cvBrightnessBar');
        if (bEl)  bEl.textContent  = pct;
        if (blEl) blEl.textContent = pct;
        if (cEl) {
            cEl.textContent = map[condition] || condition;
            cEl.className   = 'status-val' + (condition === 'normal' ? ' muted' : '');
        }
        if (bBar) bBar.style.width = (brightness * 100).toFixed(1) + '%';
    }

    // ── Overlay management ───────────────────────────────────────
    attachOverlay(containerId) {
        const cont = document.getElementById(containerId);
        if (cont && this.overlayCanvas && !cont.querySelector('#cvOverlayCanvas'))
            cont.appendChild(this.overlayCanvas);
    }

    removeOverlay() {
        if (this.overlayCanvas?.parentNode)
            this.overlayCanvas.parentNode.removeChild(this.overlayCanvas);
    }

    clearOverlay() {
        if (this.overlayContext && this.overlayCanvas)
            this.overlayContext.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    }

    // ── Loading / status ─────────────────────────────────────────
    showLoading(msg = 'Memuat…') {
        const el = document.getElementById('cvLoadingStatus');
        if (el) { el.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${msg}`; el.classList.remove('hidden'); }
    }

    hideLoading() {
        document.getElementById('cvLoadingStatus')?.classList.add('hidden');
    }

    updateSystemStatus(status) {
        const el  = document.getElementById('cvSystemStatus');
        if (!el) return;
        const map = {
            ready:    ['✅ Siap',        'ok'],
            loading:  ['⏳ Memuat…',     'muted'],
            error:    ['❌ Error',        ''],
            inactive: ['⚪ Tidak Aktif', 'muted']
        };
        const [txt, cls] = map[status] || map.inactive;
        el.textContent   = txt;
        el.className     = 'status-val ' + cls;
    }

    // ────────────────────────────────────────────────────────────
    // ── FULL AUTOMATION SETTINGS PANEL ──────────────────────────
    // ────────────────────────────────────────────────────────────
    renderAutomationSettings() {
        const cont = document.getElementById('cvAutomationSettings');
        if (!cont) return;

        if (typeof STATE === 'undefined' || typeof automationEngine === 'undefined') {
            cont.innerHTML = '<p class="cv-no-dev">Sistem belum siap.</p>';
            return;
        }

        const rules     = automationEngine.getCVRules();
        const schedules = automationEngine.getSchedules();
        const devIds    = Object.keys(STATE.devices);

        cont.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:16px">

            <!-- ── DETEKSI MANUSIA ── -->
            <div class="cv-auto-section">
                <div class="cv-auto-head">
                    <div>
                        <div class="cv-auto-title">👤 Deteksi Manusia</div>
                        <div class="cv-auto-sub">Kontrol perangkat saat ada/tidak ada orang</div>
                    </div>
                    <label class="toggle-wrapper">
                        <input type="checkbox" id="cvHumanEnabled" class="toggle-input"
                            ${rules.humanDetection.enabled ? 'checked' : ''}
                            onchange="automationEngine.setEnabled('humanDetection', this.checked)">
                        <span class="toggle-track"></span>
                    </label>
                </div>
                <div class="cv-auto-body">
                    <div>
                        <span class="cv-trigger-label">Nyalakan saat terdeteksi</span>
                        <div class="cv-device-list" id="cvOnDetectList">
                            ${this._renderDeviceCheckboxes(devIds, rules.humanDetection.onDetect, 'onDetect')}
                        </div>
                    </div>
                    <div>
                        <span class="cv-trigger-label">Matikan saat tidak ada orang</span>
                        <div class="cv-device-list" id="cvOnAbsentList">
                            ${this._renderDeviceCheckboxes(devIds, rules.humanDetection.onAbsent, 'onAbsent')}
                        </div>
                    </div>
                    <div class="cv-threshold-grid" style="margin-top:4px">
                        <div class="cv-field">
                            <label>Min. Kepercayaan</label>
                            <div class="fi-group">
                                <input type="number" id="cvConfInput" class="fi-input"
                                    value="${Math.round(CV_CONFIG.model.minConfidence * 100)}"
                                    min="10" max="99"
                                    onchange="CV_CONFIG.model.minConfidence = this.value / 100; saveCVConfig(); if(typeof CV !== 'undefined') CV.confidence = CV_CONFIG.model.minConfidence;">
                                <span class="fi-unit">%</span>
                            </div>
                        </div>
                        <div class="cv-field">
                            <label>Delay Eksekusi</label>
                            <div class="fi-group">
                                <input type="number" id="cvHumanDelay" class="fi-input"
                                    value="${rules.humanDetection.delay || 3000}"
                                    min="0" step="500">
                                <span class="fi-unit">ms</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ── KONDISI CAHAYA ── -->
            <div class="cv-auto-section">
                <div class="cv-auto-head">
                    <div>
                        <div class="cv-auto-title">💡 Kondisi Cahaya</div>
                        <div class="cv-auto-sub">Analisis kecerahan dari kamera secara real-time</div>
                    </div>
                    <label class="toggle-wrapper">
                        <input type="checkbox" id="cvLightEnabled" class="toggle-input"
                            ${rules.lightCondition.enabled ? 'checked' : ''}
                            onchange="automationEngine.setEnabled('lightCondition', this.checked)">
                        <span class="toggle-track"></span>
                    </label>
                </div>
                <div class="cv-auto-body">
                    <div>
                        <span class="cv-trigger-label">Nyalakan saat gelap</span>
                        <div class="cv-device-list" id="cvOnDarkList">
                            ${this._renderDeviceCheckboxes(devIds, rules.lightCondition.onDark, 'onDark')}
                        </div>
                    </div>
                    <div>
                        <span class="cv-trigger-label">Matikan saat terang</span>
                        <div class="cv-device-list" id="cvOnBrightList">
                            ${this._renderDeviceCheckboxes(devIds, rules.lightCondition.onBright, 'onBright')}
                        </div>
                    </div>
                    <div class="cv-threshold-grid" style="margin-top:4px">
                        <div class="cv-field">
                            <label>Ambang Gelap (&lt;)</label>
                            <div class="fi-group">
                                <input type="number" id="cvDarkThr" class="fi-input"
                                    value="${Math.round(CV_CONFIG.light.darkThreshold * 100)}"
                                    min="5" max="50"
                                    onchange="CV_CONFIG.light.darkThreshold = this.value / 100; saveCVConfig();">
                                <span class="fi-unit">%</span>
                            </div>
                        </div>
                        <div class="cv-field">
                            <label>Ambang Terang (&gt;)</label>
                            <div class="fi-group">
                                <input type="number" id="cvBrightThr" class="fi-input"
                                    value="${Math.round(CV_CONFIG.light.brightThreshold * 100)}"
                                    min="50" max="95"
                                    onchange="CV_CONFIG.light.brightThreshold = this.value / 100; saveCVConfig();">
                                <span class="fi-unit">%</span>
                            </div>
                        </div>
                        <div class="cv-field">
                            <label>Delay Eksekusi</label>
                            <div class="fi-group">
                                <input type="number" id="cvLightDelay" class="fi-input"
                                    value="${rules.lightCondition.delay || 2000}"
                                    min="0" step="500">
                                <span class="fi-unit">ms</span>
                            </div>
                        </div>
                        <div class="cv-field">
                            <label>Interval Analisis</label>
                            <div class="fi-group">
                                <input type="number" id="cvLightInterval" class="fi-input"
                                    value="${CV_CONFIG.light.analysisInterval}"
                                    min="200" max="5000" step="200"
                                    onchange="CV_CONFIG.light.analysisInterval = parseInt(this.value); saveCVConfig(); if(typeof lightAnalyzer !== 'undefined') lightAnalyzer.restartWithNewInterval();">
                                <span class="fi-unit">ms</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ── JADWAL ── -->
            <div class="cv-auto-section">
                <div class="cv-auto-head">
                    <div>
                        <div class="cv-auto-title">⏰ Jadwal Otomatis</div>
                        <div class="cv-auto-sub">Aktifkan perangkat berdasarkan waktu &amp; hari</div>
                    </div>
                    <button onclick="cvUI.openAddScheduleModal()" class="btn-ghost small">
                        <i class="fas fa-plus"></i> Tambah
                    </button>
                </div>
                <div class="cv-auto-body" id="cvScheduleList">
                    ${this._renderSchedules(schedules)}
                </div>
            </div>

            <!-- ── SAVE ── -->
            <button onclick="cvUI.saveAutomationSettings()" class="btn-primary"
                style="width:100%;justify-content:center">
                <i class="fas fa-floppy-disk"></i> Simpan Semua Pengaturan CV
            </button>
        </div>`;
    }

    // ── Device checkbox list ─────────────────────────────────────
    _renderDeviceCheckboxes(devIds, selected, category) {
        const esc = typeof escHtml === 'function' ? escHtml : (s) => String(s || '');
        if (!devIds.length)
            return '<div class="cv-no-dev">Belum ada perangkat ditambahkan</div>';
        return devIds.map(id => {
            const dev  = STATE.devices[id];
            if (!dev) return '';
            const isCh = (selected || []).map(String).includes(String(id));
            return `
            <label class="cv-dev-cb-item${isCh ? ' checked' : ''}">
                <input type="checkbox" class="cv-dev-cb" data-cat="${category}" value="${id}"
                    ${isCh ? 'checked' : ''}
                    onchange="this.closest('label').classList.toggle('checked', this.checked)">
                <i class="fas ${dev.icon || 'fa-plug'}"></i>
                <span>${esc(dev.name)}</span>
            </label>`;
        }).join('');
    }

    // ── Schedule list ────────────────────────────────────────────
    _renderSchedules(schedules) {
        if (!schedules.length) {
            return '<div class="cv-no-dev">Belum ada jadwal. Klik "+ Tambah" untuk membuat jadwal.</div>';
        }
        const dayNames = ['Min', 'Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab'];
        const esc      = typeof escHtml === 'function' ? escHtml : (s) => String(s || '');
        return schedules.map(s => {
            const devNames = (s.devices || [])
                .map(id => STATE.devices[String(id)]?.name || `#${id}`)
                .join(', ') || '—';
            const daysStr   = s.days?.length ? s.days.map(d => dayNames[d]).join(', ') : 'Setiap hari';
            const actionBadge = s.action === 'off'
                ? `<span class="rule-badge-off">OFF</span>`
                : `<span class="rule-badge-on">ON</span>`;
            return `
            <div class="rule-row" id="sched-row-${s.id}">
                <label class="toggle-wrapper" style="flex-shrink:0">
                    <input type="checkbox" class="toggle-input" ${s.enabled ? 'checked' : ''}
                        onchange="automationEngine.toggleSchedule('${s.id}', this.checked)">
                    <span class="toggle-track"></span>
                </label>
                <div style="flex:1;min-width:0">
                    <div style="font-size:12px;font-weight:600;color:var(--ink);display:flex;align-items:center;gap:6px">
                        <span style="font-family:var(--mono)">${esc(s.time)}</span>
                        ${actionBadge}
                        <span style="font-size:11px;color:var(--ink-3)">${esc(devNames)}</span>
                    </div>
                    <div style="font-size:10.5px;color:var(--ink-4);margin-top:2px">${esc(daysStr)}</div>
                </div>
                <button onclick="cvUI.removeSchedule('${s.id}')" class="trash-btn" title="Hapus">
                    <i class="fas fa-trash"></i>
                </button>
            </div>`;
        }).join('');
    }

    // ── Save all CV automation settings ─────────────────────────
    saveAutomationSettings() {
        const getList = cat =>
            Array.from(document.querySelectorAll(`.cv-dev-cb[data-cat="${cat}"]:checked`))
                 .map(c => c.value);

        const humanDelay = parseInt(document.getElementById('cvHumanDelay')?.value) || 3000;
        const lightDelay = parseInt(document.getElementById('cvLightDelay')?.value) || 2000;

        automationEngine.updateCVRules({
            humanDetection: {
                onDetect: getList('onDetect'),
                onAbsent: getList('onAbsent'),
                delay:    humanDelay
            },
            lightCondition: {
                onDark:   getList('onDark'),
                onBright: getList('onBright'),
                delay:    lightDelay
            }
        });

        // Keep CV runtime object in sync
        if (typeof CV !== 'undefined') {
            CV.cvRules.human.onDetect = getList('onDetect');
            CV.cvRules.human.onAbsent = getList('onAbsent');
            CV.cvRules.human.delay    = humanDelay;
            CV.cvRules.light.onDark   = getList('onDark');
            CV.cvRules.light.onBright = getList('onBright');
            CV.cvRules.light.delay    = lightDelay;
            if (typeof saveCVRules === 'function') saveCVRules();
        }

        if (typeof showToast === 'function') showToast('✅ Pengaturan CV disimpan!', 'success');
    }

    // ── Schedule modal ────────────────────────────────────────────
    openAddScheduleModal() {
        const esc     = typeof escHtml === 'function' ? escHtml : (s) => String(s || '');
        const devOpts = Object.entries(STATE.devices)
            .map(([id, d]) => `<option value="${id}">${esc(d.name)}</option>`)
            .join('');
        const dayLabels = ['Min', 'Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab'];
        const dayChecks = dayLabels.map((l, i) => `
            <label style="display:flex;align-items:center;gap:4px;font-size:11.5px;cursor:pointer">
                <input type="checkbox" class="sched-day" value="${i}"
                    ${i > 0 && i < 6 ? 'checked' : ''}
                    style="accent-color:var(--a)">
                ${l}
            </label>`).join('');

        // Remove existing modal if any
        document.getElementById('cvScheduleModal')?.remove();

        const modal = document.createElement('div');
        modal.className = 'modal-backdrop show';
        modal.id        = 'cvScheduleModal';
        modal.innerHTML = `
            <div class="modal">
                <div class="modal-head">
                    <h3 class="modal-title">Tambah Jadwal</h3>
                    <button onclick="document.getElementById('cvScheduleModal').remove()"
                        class="modal-close"><i class="fas fa-times"></i></button>
                </div>
                <div class="modal-body">
                    <div class="field-group">
                        <label>Waktu</label>
                        <input type="time" id="schedTime" class="field-input" value="18:00">
                    </div>
                    <div class="field-group">
                        <label>Hari</label>
                        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px">${dayChecks}</div>
                    </div>
                    <div class="field-group">
                        <label>Perangkat</label>
                        <select id="schedDevices" multiple class="field-input" style="height:90px">
                            ${devOpts || '<option disabled>Belum ada perangkat</option>'}
                        </select>
                        <span style="font-size:10.5px;color:var(--ink-4);margin-top:3px">Tahan Ctrl/Cmd untuk pilih banyak</span>
                    </div>
                    <div class="field-group">
                        <label>Aksi</label>
                        <select id="schedAction" class="field-input">
                            <option value="on">Nyalakan (ON)</option>
                            <option value="off">Matikan (OFF)</option>
                        </select>
                    </div>
                    <div class="field-group">
                        <label>Label <span style="font-size:10px;color:var(--ink-5)">(opsional)</span></label>
                        <input type="text" id="schedLabel" class="field-input" placeholder="cth: Lampu malam">
                    </div>
                </div>
                <div class="modal-footer">
                    <button onclick="document.getElementById('cvScheduleModal').remove()"
                        class="btn-ghost">Batal</button>
                    <button onclick="cvUI.saveSchedule()" class="btn-primary">
                        <i class="fas fa-plus"></i> Simpan Jadwal
                    </button>
                </div>
            </div>`;
        document.body.appendChild(modal);
    }

    saveSchedule() {
        const time    = document.getElementById('schedTime')?.value;
        const days    = Array.from(document.querySelectorAll('.sched-day:checked')).map(c => parseInt(c.value));
        const selEl   = document.getElementById('schedDevices');
        const devices = Array.from(selEl?.selectedOptions || []).map(o => o.value);
        const action  = document.getElementById('schedAction')?.value || 'on';
        const label   = document.getElementById('schedLabel')?.value?.trim();

        if (!time)           { if (typeof showToast === 'function') showToast('Pilih waktu!', 'warning');                   return; }
        if (!devices.length) { if (typeof showToast === 'function') showToast('Pilih minimal 1 perangkat!', 'warning');     return; }

        automationEngine.addSchedule(time, days, devices, action, label);
        document.getElementById('cvScheduleModal')?.remove();
        this.renderAutomationSettings();
        if (typeof showToast === 'function') showToast('Jadwal ditambahkan!', 'success');
    }

    removeSchedule(id) {
        automationEngine.removeSchedule(id);
        this.renderAutomationSettings();
        if (typeof showToast === 'function') showToast('Jadwal dihapus', 'info');
    }

    destroy() { this.removeOverlay(); }
}

const cvUI = new CVUI();
