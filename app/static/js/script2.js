document.addEventListener('DOMContentLoaded', () => {

  // ===============================
  // ELEMENTS
  // ===============================
  const feeds = Array.from(document.querySelectorAll('video[id^="feed"]'));
  const globalStatusText = document.getElementById('global-status-text');
  const globalStatusMsg = document.getElementById('global-status-msg');
  const themeToggle = document.getElementById('theme-toggle');
  const refreshBtn = document.getElementById('refresh-btn');
  const exportBtn = document.getElementById('export-btn');

  let detectionLog = [];

  // ===============================
  // DARK MODE TOGGLE
  // ===============================
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('dark-mode');

      const icon = themeToggle.querySelector('i');
      if (document.body.classList.contains('dark-mode')) {
        icon.classList.replace('fa-moon', 'fa-sun');
      } else {
        icon.classList.replace('fa-sun', 'fa-moon');
      }
    });
  }

  // ===============================
  // REFRESH BUTTON
  // ===============================
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      console.log('🔄 Manual refresh triggered');
      analyzeAllFeeds();
    });
  }

  // ===============================
  // EXPORT LOGS BUTTON
  // ===============================
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (detectionLog.length === 0) {
        alert('No detection logs to export');
        return;
      }

      const logText = detectionLog.map(log => {
        return `[${log.timestamp}] Camera ${log.camera}: ${log.status} - ${log.details}`;
      }).join('\n');

      const blob = new Blob([logText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `crime_detection_log_${new Date().toISOString().slice(0,10)}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // ===============================
  // TIME UPDATE (BOTTOM RIGHT)
  // ===============================
  function updateTime() {
    const now = new Date();
    const time = now.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });

    document.querySelectorAll('.feed-footer').forEach(el => {
      el.textContent = time;
    });
  }

  // ===============================
  // ANALYZE ONE FEED
  // ===============================
  async function analyzeFeed(videoEl) {
    const src =
      videoEl.querySelector('source')?.getAttribute('src') ||
      videoEl.getAttribute('src');

    if (!src) return { threat: false, details: '' };

    const videoPath = src.replace(/^\/?static\//, '');

    try {
      const res = await fetch('/analyze_video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoPath })
      });

      if (!res.ok) return { threat: false, details: 'Analysis failed' };

      const data = await res.json();
      const feedBox = videoEl.closest('.feed-box');
      const statusEl = feedBox.querySelector('.status');
      const cameraId = videoEl.id;
      const cameraNum = cameraId.replace('feed', '');

      // ===============================
      // CAMERA STATUS + BLINK
      // ===============================
      if (data.final_threat) {
        const crimeType = data.crime_type || 'Unknown';
        const confidence = (data.crime_confidence * 100).toFixed(1);
        
        statusEl.textContent = `🚨 ${crimeType.toUpperCase()}`;
        statusEl.className = 'status danger';
        statusEl.title = `Confidence: ${confidence}%`;

        // 🔴 blinking red border
        feedBox.classList.add('threat');

        // Add to detection log
        detectionLog.push({
          timestamp: new Date().toLocaleString(),
          camera: cameraNum,
          status: 'THREAT DETECTED',
          details: `${crimeType} (${confidence}% confidence)`
        });

        // Show crime details overlay
        let crimeOverlay = feedBox.querySelector('.crime-details');
        if (!crimeOverlay) {
          crimeOverlay = document.createElement('div');
          crimeOverlay.className = 'crime-details';
          feedBox.appendChild(crimeOverlay);
        }
        crimeOverlay.innerHTML = `
          <strong>${crimeType}</strong><br>
          Confidence: ${confidence}%
        `;

        return { threat: true, details: `${crimeType} (${confidence}%)` };

      } else {
        statusEl.textContent = 'MONITORING';
        statusEl.className = 'status safe';
        statusEl.title = 'Normal activity';

        // remove blink
        feedBox.classList.remove('threat');

        // Remove crime overlay
        const crimeOverlay = feedBox.querySelector('.crime-details');
        if (crimeOverlay) {
          crimeOverlay.remove();
        }
      }

      // ===============================
      // TRANSCRIPTION OVERLAY
      // ===============================
      let overlay = feedBox.querySelector('.transcription');
      if (data.transcription && data.transcription.trim()) {
        if (!overlay) {
          overlay = document.createElement('div');
          overlay.className = 'transcription';
          feedBox.appendChild(overlay);
        }
        overlay.textContent = `🎤 ${data.transcription}`;
      } else if (overlay) {
        overlay.remove();
      }

      return { threat: false, details: 'Normal activity' };

    } catch (err) {
      console.error('Analysis error:', err);
      return { threat: false, details: 'Error' };
    }
  }

  // ===============================
  // ANALYZE ALL FEEDS
  // ===============================
  async function analyzeAllFeeds() {
    let anyThreat = false;
    let threatDetails = [];

    for (const video of feeds) {
      const result = await analyzeFeed(video);
      if (result.threat) {
        anyThreat = true;
        const cameraNum = video.id.replace('feed', '');
        threatDetails.push(`Camera ${cameraNum}: ${result.details}`);
      }
    }

    // ===============================
    // GLOBAL STATUS
    // ===============================
    if (anyThreat) {
      globalStatusText.textContent = '🚨 ALERT';
      globalStatusText.className = 'status danger';
      globalStatusMsg.innerHTML = `
        <strong>Crime detected in ${threatDetails.length} camera(s):</strong><br>
        ${threatDetails.join('<br>')}
      `;
    } else {
      globalStatusText.textContent = 'MONITORING';
      globalStatusText.className = 'status safe';
      globalStatusMsg.textContent = 'All systems operational';
    }
  }

  // ===============================
  // SIDEBAR NAVIGATION
  // ===============================
  const sidebarItems = document.querySelectorAll('.sidebar li:not(#theme-toggle)');
  sidebarItems.forEach(item => {
    item.addEventListener('click', function() {
      sidebarItems.forEach(i => i.classList.remove('active'));
      this.classList.add('active');
      
      const text = this.textContent.trim();
      if (text.includes('Home')) {
        document.querySelector('.live-feeds').style.display = 'block';
        document.querySelector('.threats-overview').style.display = 'block';
      } else if (text.includes('Live Feeds')) {
        document.querySelector('.live-feeds').style.display = 'block';
        document.querySelector('.threats-overview').style.display = 'none';
      } else if (text.includes('Logs')) {
        alert(`Detection Logs:\n\n${detectionLog.length === 0 ? 'No logs yet' : detectionLog.map(l => `[${l.timestamp}] Camera ${l.camera}: ${l.details}`).join('\n')}`);
      } else if (text.includes('Settings')) {
        alert('Settings panel coming soon!');
      }
    });
  });

  // ===============================
  // INIT
  // ===============================
  updateTime();
  analyzeAllFeeds();

  setInterval(updateTime, 60000);        // clock
  setInterval(analyzeAllFeeds, 20000);   // AI polling every 20s
});