const CV_CONFIG = {
  model: { minConfidence: 0.65, maxBoxes: 20 },
  ui: { showBoundingBoxes: true, showDebugInfo: true }
};
function loadCVConfig(){
  try{ const s = localStorage.getItem('iotzy_cv_config'); if(s){ Object.assign(CV_CONFIG, JSON.parse(s)); } }catch(_){ }
}
function saveCVConfig(){
  try{ localStorage.setItem('iotzy_cv_config', JSON.stringify(CV_CONFIG)); }catch(_){ }
}
