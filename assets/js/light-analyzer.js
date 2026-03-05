const lightAnalyzer = {
  running:false, timer:null,
  startAnalysis(video){
    if (this.running) return; this.running = true;
    const c = document.createElement('canvas'); const x = c.getContext('2d', {willReadFrequently:true});
    this.timer = setInterval(() => {
      if (!this.running || !video.videoWidth) return;
      c.width = 64; c.height = 48; x.drawImage(video,0,0,c.width,c.height);
      const data = x.getImageData(0,0,c.width,c.height).data;
      let sum=0; for(let i=0;i<data.length;i+=4) sum += (0.2126*data[i]+0.7152*data[i+1]+0.0722*data[i+2]);
      const brightness = Math.round(sum / (data.length/4));
      document.getElementById('cvBrightness') && (document.getElementById('cvBrightness').textContent = String(brightness));
    }, 500);
  },
  stopAnalysis(){ this.running=false; if(this.timer) clearInterval(this.timer); this.timer=null; }
};
