// ==================== LIGHT ANALYZER MODULE ====================
class LightAnalyzer {
    constructor() {
        this.canvas          = null;
        this.ctx             = null;
        this.isActive        = false;
        this.interval        = null;
        this.lastBrightness  = null;
        this.lastCondition   = null;
        this.videoElement    = null;

        // smoothing + hysteresis agar stabil & smooth.
        this.smoothedBrightness = null;
        this.smoothingAlpha     = 0.35;
        this.hysteresisMargin   = 0.02;

        // Multi-callback support (same pattern as CVDetector)
        this._callbacks = {
            onLightChange:      [],
            onBrightnessUpdate: []
        };
    }

    startAnalysis(videoElement) {
        if (this.isActive) this.stopAnalysis(); // restart cleanly if already running

        this.videoElement = videoElement;
        this.isActive     = true;

        const size         = CV_CONFIG.light.sampleSize || 80;
        this.canvas        = document.createElement('canvas');
        this.canvas.width  = size;
        this.canvas.height = size;
        this.ctx           = this.canvas.getContext('2d', { willReadFrequently: true });

        this.smoothedBrightness = null;
        this._startInterval();
        console.log('💡 Light analyzer dimulai');
    }

    _startInterval() {
        if (this.interval) clearInterval(this.interval);
        const ms      = CV_CONFIG.light.analysisInterval || 1000;
        this.interval = setInterval(() => this._analyze(), ms);
    }

    /**
     * Call this when CV_CONFIG.light.analysisInterval is changed at runtime
     * so the new interval takes effect immediately.
     */
    restartWithNewInterval() {
        if (!this.isActive) return;
        this._startInterval();
    }

    stopAnalysis() {
        if (this.interval) { clearInterval(this.interval); this.interval = null; }
        this.isActive          = false;
        this.lastCondition     = null;
        this.lastBrightness    = null;
        this.smoothedBrightness = null;
        console.log('💡 Light analyzer dihentikan');
    }

    _analyze() {
        const v = this.videoElement;
        if (!v || v.readyState < 2 || v.paused || v.videoWidth < 1) return;
        try {
            const W = this.canvas.width;
            const H = this.canvas.height;
            this.ctx.drawImage(v, 0, 0, W, H);
            const px    = this.ctx.getImageData(0, 0, W, H).data;
            let total   = 0;
            let count   = 0;

            // Sample every 4th pixel (step 16 bytes = 4 channels × 4 pixels) for performance
            for (let i = 0; i < px.length; i += 16) {
                total += (0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]) / 255;
                count++;
            }

            const rawBrightness = count ? total / count : 0;
            if (this.smoothedBrightness == null) {
                this.smoothedBrightness = rawBrightness;
            } else {
                this.smoothedBrightness =
                    (this.smoothingAlpha * rawBrightness) +
                    ((1 - this.smoothingAlpha) * this.smoothedBrightness);
            }

            const brightness = this.smoothedBrightness;
            const bright     = CV_CONFIG.light.brightThreshold;
            const dark       = CV_CONFIG.light.darkThreshold;

            const cond = this._classifyWithHysteresis(brightness, bright, dark);
            this._emit('onBrightnessUpdate', brightness, cond);

            if (cond !== this.lastCondition) {
                this.lastCondition = cond;
                this._emit('onLightChange', cond, brightness);
            }
            this.lastBrightness = brightness;
        } catch (_) {}
    }

    _classifyWithHysteresis(brightness, bright, dark) {
        const margin = this.hysteresisMargin;
        const prev   = this.lastCondition;

        if (prev === 'dark') {
            if (brightness > dark + margin) {
                return brightness >= bright ? 'bright' : 'normal';
            }
            return 'dark';
        }

        if (prev === 'bright') {
            if (brightness < bright - margin) {
                return brightness <= dark ? 'dark' : 'normal';
            }
            return 'bright';
        }

        if (brightness >= bright) return 'bright';
        if (brightness <= dark)   return 'dark';
        return 'normal';
    }

    // ── Emit helpers (multi-callback, consistent with CVDetector) ────────────
    _emit(event, ...args) {
        const list = this._callbacks[event];
        if (!list) return;
        list.forEach(fn => {
            try { fn(...args); } catch (e) { console.warn('LightAnalyzer CB error', event, e); }
        });
    }

    /**
     * setCallbacks — MERGES new callbacks instead of replacing.
     * Uses cb._tag to allow replacing a specific group (same as CVDetector).
     */
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
        if (cb.onLightChange)      addOrReplace('onLightChange',      cb.onLightChange);
        if (cb.onBrightnessUpdate) addOrReplace('onBrightnessUpdate', cb.onBrightnessUpdate);
    }

    getBrightness() { return this.lastBrightness; }
    getCondition()  { return this.lastCondition; }

    destroy() { this.stopAnalysis(); }
}

const lightAnalyzer = new LightAnalyzer();
