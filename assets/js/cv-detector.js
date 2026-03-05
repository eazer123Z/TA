// ==================== CV DETECTOR MODULE (Enhanced) ====================
class CVDetector {
    constructor() {
        this.model             = null;
        this.isLoading         = false;
        this.isReady           = false;
        this.detectionActive   = false;
        this.detectionInterval = null;
        this.lastDetectionTime = 0;
        this.detectionHistory  = [];
        this.currentDetections = [];
        this.humanPresent      = false;
        this.lastHumanState    = false;
        this.lastPersonCount   = 0;
        this.presenceTimer     = null;
        this.loadAttempts      = 0;
        this.MAX_ATTEMPTS      = 3;
        this.videoElement      = null;
        this._callbacks = {
            onHumanDetected:    [],
            onHumanAbsent:      [],
            onDetectionUpdate:  [],
            onPersonCountChange:[],
            onError:            []
        };
    }

    _waitForLibraries(maxWait = 20000) {
        return new Promise((resolve, reject) => {
            const start = Date.now();
            const check = () => {
                const tfOk = typeof tf !== 'undefined' && typeof tf.ready === 'function';
                const sdOk = typeof cocoSsd !== 'undefined';
                if (tfOk && sdOk) { resolve(); return; }
                if (Date.now() - start > maxWait) {
                    reject(new Error(
                        !tfOk
                            ? 'TensorFlow.js tidak tersedia. Periksa koneksi internet.'
                            : 'Model COCO-SSD tidak tersedia. Periksa koneksi internet.'
                    ));
                    return;
                }
                setTimeout(check, 300);
            };
            check();
        });
    }

    async initialize() {
        if (this.isLoading)             { return false; }
        if (this.isReady && this.model) { return true;  }

        this.isLoading = true;
        this.loadAttempts++;
        console.log(`⏳ CV: Memuat model (percobaan ${this.loadAttempts})…`);

        try {
            await this._waitForLibraries();
            await tf.ready();
            console.log(`✅ TF Backend: ${tf.getBackend()}`);

            this.model = await cocoSsd.load({
                base: CV_CONFIG.model.base || 'lite_mobilenet_v2'
            });

            const dummy = tf.zeros([1, 100, 100, 3]);
            try { await this.model.detect(dummy); } catch (_) {}
            dummy.dispose();

            this.isReady      = true;
            this.isLoading    = false;
            this.loadAttempts = 0;
            console.log('✅ CV Detector siap!');
            return true;

        } catch (err) {
            console.error('❌ CV model error:', err.message || err);
            this.isLoading = false;
            this.model     = null;
            this.isReady   = false;
            this._emit('onError', err.message || String(err));
            return false;
        }
    }

    canRetry() { return this.loadAttempts < this.MAX_ATTEMPTS; }

    startDetection(videoElement) {
        if (!this.isReady || !this.model) { return false; }
        if (this.detectionActive)          { return true; }

        this.videoElement      = videoElement;
        this.detectionActive   = true;
        this.lastDetectionTime = 0;

        const interval         = CV_CONFIG.detection.interval || 500;
        this.detectionInterval = setInterval(() => this._runDetection(), interval);
        return true;
    }

    stopDetection() {
        if (this.detectionInterval) { clearInterval(this.detectionInterval); this.detectionInterval = null; }
        if (this.presenceTimer)     { clearTimeout(this.presenceTimer);       this.presenceTimer     = null; }
        this.detectionActive   = false;
        this.currentDetections = [];
        this.humanPresent      = false;
        this.lastHumanState    = false;
        this.lastPersonCount   = 0;
    }

    async _runDetection() {
        if (!this.detectionActive || !this.model || !this.videoElement) return;
        const v = this.videoElement;
        if (!v || v.readyState < 2 || v.paused || v.videoWidth < 1) return;

        const now = Date.now();
        if (now - this.lastDetectionTime < (CV_CONFIG.detection.interval - 50)) return;
        this.lastDetectionTime = now;

        try {
            const preds   = await this.model.detect(v);
            const minConf = CV_CONFIG.model.minConfidence || 0.6;
            const humans  = preds.filter(p => p.class === 'person' && p.score >= minConf);

            this.currentDetections = humans;
            this._updatePresence(humans.length > 0);

            // Person count change notification
            if (humans.length !== this.lastPersonCount) {
                this.lastPersonCount = humans.length;
                this._emit('onPersonCountChange', humans.length);
                // Also call global app function if available
                if (typeof onCVPersonCountUpdate === 'function') {
                    onCVPersonCountUpdate(humans.length);
                }
            }

            if (typeof CV !== 'undefined') CV.frameCount++;

            this._emit('onDetectionUpdate', {
                detections:    humans,
                humanCount:    humans.length,
                humanPresent:  this.humanPresent,
                avgConfidence: humans.length
                    ? humans.reduce((s, d) => s + d.score, 0) / humans.length
                    : 0,
                timestamp: now
            });
        } catch (err) {
            console.warn('Detection frame error:', err.message || err);
        }
    }

    _updatePresence(detected) {
        if (this.presenceTimer) clearTimeout(this.presenceTimer);

        this.detectionHistory.push({ detected, ts: Date.now() });
        const cutoff          = Date.now() - 5000;
        this.detectionHistory = this.detectionHistory.filter(h => h.ts > cutoff);

        const debounce    = CV_CONFIG.detection.debounceTime || 1500;
        this.presenceTimer = setTimeout(() => {
            if (detected === this.lastHumanState) return;
            this.lastHumanState = detected;
            this.humanPresent   = detected;
            if (detected) this._emit('onHumanDetected');
            else          this._emit('onHumanAbsent');
        }, debounce);
    }

    _emit(event, data) {
        const list = this._callbacks[event];
        if (!list) return;
        list.forEach(fn => {
            try { fn(data); } catch (e) { console.warn('CVDetector CB error', event, e); }
        });
    }

    setCallbacks(cb) {
        const addOrReplace = (key, fn) => {
            if (typeof fn !== 'function') return;
            const tag  = cb._tag || null;
            const list = this._callbacks[key] || (this._callbacks[key] = []);
            if (tag) {
                const tagged = fn;
                tagged._tag  = tag;
                const idx    = list.findIndex(f => f._tag === tag);
                if (idx >= 0) list[idx] = tagged;
                else          list.push(tagged);
            } else {
                list.push(fn);
            }
        };
        if (cb.onHumanDetected)    addOrReplace('onHumanDetected',    cb.onHumanDetected);
        if (cb.onHumanAbsent)      addOrReplace('onHumanAbsent',      cb.onHumanAbsent);
        if (cb.onDetectionUpdate)  addOrReplace('onDetectionUpdate',  cb.onDetectionUpdate);
        if (cb.onPersonCountChange)addOrReplace('onPersonCountChange', cb.onPersonCountChange);
        if (cb.onError)            addOrReplace('onError',            cb.onError);
    }

    getDetections()  { return this.currentDetections; }
    getHumanCount()  { return this.currentDetections.length; }
    isHumanPresent() { return this.humanPresent; }
    getStatus()      { return this.isLoading ? 'loading' : this.isReady ? 'ready' : 'idle'; }

    destroy() {
        this.stopDetection();
        this.model   = null;
        this.isReady = false;
    }
}

const cvDetector = new CVDetector();
