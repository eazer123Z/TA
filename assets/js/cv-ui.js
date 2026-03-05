const cvUI = {
  overlay:null, ctx:null,
  initialize(){ this.attachOverlay('cameraFocusContainer'); },
  attachOverlay(containerId){
    const c = document.getElementById('cvOverlayCanvas');
    this.overlay = c; this.ctx = c?.getContext('2d');
  },
  clearOverlay(){ if(!this.ctx||!this.overlay) return; this.ctx.clearRect(0,0,this.overlay.width,this.overlay.height); },
  drawDetections(persons){
    if(!this.overlay||!this.ctx) return;
    const v = document.getElementById('cameraFocus');
    if(!v?.videoWidth) return;
    this.overlay.width = v.clientWidth; this.overlay.height = v.clientHeight;
    this.ctx.clearRect(0,0,this.overlay.width,this.overlay.height);
    if (!CV_CONFIG.ui.showBoundingBoxes) return;
    this.ctx.strokeStyle = '#22c55e'; this.ctx.lineWidth = 2;
    persons.forEach(p=>{
      const [x,y,w,h] = p.bbox;
      const sx = x*(this.overlay.width/v.videoWidth), sy = y*(this.overlay.height/v.videoHeight);
      const sw = w*(this.overlay.width/v.videoWidth), sh = h*(this.overlay.height/v.videoHeight);
      this.ctx.strokeRect(sx,sy,sw,sh);
    });
  },
  renderAutomationSettings(){
    const el = document.getElementById('cvAutomationSettings');
    if(!el) return;
    el.innerHTML = '<p>Human detection: coco-ssd mobilenet_v2, confidence '+Math.round(CV_CONFIG.model.minConfidence*100)+'%</p>';
  }
};
