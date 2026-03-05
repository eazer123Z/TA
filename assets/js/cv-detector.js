const cvDetector = {
  model: null,
  running: false,
  rafId: null,
  callbacks: {},
  setCallbacks(cb){ this.callbacks = cb || {}; },
  async initialize(){
    try{
      await tf.ready();
      const backendOrder = ['webgl','wasm','cpu'];
      for (const b of backendOrder){ try{ await tf.setBackend(b); await tf.ready(); break; }catch(_){} }
      this.model = await cocoSsd.load({ base: 'mobilenet_v2' });
      return true;
    }catch(e){ this.callbacks.onError?.(e.message || String(e)); return false; }
  },
  startDetection(video){
    if (!this.model || this.running) return;
    this.running = true;
    const loop = async () => {
      if (!this.running) return;
      try {
        const preds = await this.model.detect(video, CV_CONFIG.model.maxBoxes, CV_CONFIG.model.minConfidence);
        const persons = preds.filter(p => p.class === 'person');
        window.onCVPersonCountUpdate?.(persons.length);
        window.cvUI?.drawDetections?.(persons);
      } catch(e){ this.callbacks.onError?.(e.message || String(e)); }
      this.rafId = requestAnimationFrame(loop);
    };
    loop();
  },
  stopDetection(){ this.running = false; if (this.rafId) cancelAnimationFrame(this.rafId); this.rafId = null; }
};
