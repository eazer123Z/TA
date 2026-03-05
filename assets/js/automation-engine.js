const automationEngine = {
  isActive:false,
  initialize(){},
  start(){this.isActive=true;},
  stop(){this.isActive=false;},
  evaluateSensorRules(sensorId, value){
    if(!this.isActive || !window.STATE?.automationRules) return;
    const rules = STATE.automationRules[String(sensorId)] || [];
    rules.forEach(r => {
      if(!r.enabled) return;
      let hit=false;
      if(r.condition==='gt') hit = value > Number(r.threshold);
      if(r.condition==='lt') hit = value < Number(r.threshold);
      if(r.condition==='detected') hit = !!value;
      if(r.condition==='absent') hit = !value;
      if(hit){
        const ns = r.action === 'on';
        window.applyDeviceState?.(String(r.deviceId), ns, `Auto Sensor ${sensorId}`);
      }
    });
  },
  getCVRules(){ return {humanDetection:{onDetect:[],onAbsent:[],delay:2000},lightCondition:{onDark:[],onBright:[],delay:2000}}; }
};
