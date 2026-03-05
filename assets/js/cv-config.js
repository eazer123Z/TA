// ==================== CV CONFIGURATION ====================
const CV_CONFIG = {
    model: {
        name:          'coco-ssd',
        base:          'lite_mobilenet_v2',
        minConfidence: 0.60,
        maxDetections: 20,
        targetClass:   'person'
    },
    detection: {
        enabled:      true,
        interval:     500,
        cooldown:     2000,
        debounceTime: 1500
    },
    light: {
        enabled:          true,
        sampleSize:       80,
        brightThreshold:  0.62,
        darkThreshold:    0.28,
        analysisInterval: 1000
    },
    camera: {
        defaultConstraints: {
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }
        },
        availableDevices: []
    },
    automation: {
        humanDetection: { enabled: true, onDetect: [], onAbsent: [], delay: 3000 },
        lightCondition: { enabled: true, onDark: [],   onBright: [], delay: 2000 }
    },
    ui: {
        showDebugInfo:     true,
        showBoundingBoxes: true,
        confidenceDisplay: true,
        overlayColor:      '#6366f1'
    }
};

function saveCVConfig() {
    try {
        localStorage.setItem('iotzy_cv_config', JSON.stringify({
            minConfidence:     CV_CONFIG.model.minConfidence,
            brightThreshold:   CV_CONFIG.light.brightThreshold,
            darkThreshold:     CV_CONFIG.light.darkThreshold,
            detectionInterval: CV_CONFIG.detection.interval,
            debounceTime:      CV_CONFIG.detection.debounceTime,
            analysisInterval:  CV_CONFIG.light.analysisInterval,
            showBoundingBoxes: CV_CONFIG.ui.showBoundingBoxes,
            showDebugInfo:     CV_CONFIG.ui.showDebugInfo
        }));
    } catch (_) {}
}

function loadCVConfig() {
    try {
        const raw = localStorage.getItem('iotzy_cv_config');
        if (!raw) return;
        const c = JSON.parse(raw);
        if (c.minConfidence     != null) CV_CONFIG.model.minConfidence    = parseFloat(c.minConfidence);
        if (c.brightThreshold   != null) CV_CONFIG.light.brightThreshold  = parseFloat(c.brightThreshold);
        if (c.darkThreshold     != null) CV_CONFIG.light.darkThreshold    = parseFloat(c.darkThreshold);
        if (c.detectionInterval != null) CV_CONFIG.detection.interval     = parseInt(c.detectionInterval);
        if (c.debounceTime      != null) CV_CONFIG.detection.debounceTime = parseInt(c.debounceTime);
        if (c.analysisInterval  != null) CV_CONFIG.light.analysisInterval = parseInt(c.analysisInterval);
        if (c.showBoundingBoxes != null) CV_CONFIG.ui.showBoundingBoxes   = !!c.showBoundingBoxes;
        if (c.showDebugInfo     != null) CV_CONFIG.ui.showDebugInfo       = !!c.showDebugInfo;
    } catch (_) {}
    if (typeof CV !== 'undefined') {
        CV.confidence = CV_CONFIG.model.minConfidence;
        CV.showBoxes  = CV_CONFIG.ui.showBoundingBoxes;
        CV.showDebug  = CV_CONFIG.ui.showDebugInfo;
    }
}
