document.addEventListener('DOMContentLoaded', function() {
  // Dark mode toggle
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', function() {
      document.body.classList.toggle('dark-mode');
      
      // Update icon
      const icon = this.querySelector('i');
      if (document.body.classList.contains('dark-mode')) {
        icon.classList.replace('fa-moon', 'fa-sun');
      } else {
        icon.classList.replace('fa-sun', 'fa-moon');
      }
      
      // Save preference
      localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    });
    
    // Initialize from saved preference
    if (localStorage.getItem('darkMode') === 'true') {
      document.body.classList.add('dark-mode');
      themeToggle.querySelector('i').classList.replace('fa-moon', 'fa-sun');
    }
  }
  
  // Add to DOMContentLoaded event
const statusSocket = new WebSocket(`ws://${window.location.host}/ws/status`);

statusSocket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    // Update threat status
    if (data.threat_detected) {
        document.querySelector('#feed1 .status').textContent = 'THREAT DETECTED';
        document.querySelector('#feed1 .status').className = 'status danger';
    }
    
    // Update timestamps
    document.querySelectorAll('.feed-footer').forEach(el => {
        el.textContent = data.timestamp;
    });
};

  // Update timestamps every minute
  function updateTimestamps() {
    const now = new Date();
    const timeString = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    document.querySelectorAll('.feed-footer').forEach(el => {
      el.textContent = timeString;
    });
  }
  
  setInterval(updateTimestamps, 60000);
  updateTimestamps(); // Initial update

  // --- Multimodal analysis integration ---
  // Collect feed video elements and map to friendly ids
  const feeds = Array.from(document.querySelectorAll('video[id^="feed"]'));

  async function analyzeFeed(videoEl) {
    try {
      // Extract the static video path used by the backend. url looks like '/static/videos/feed1.mp4'
      const src = videoEl.querySelector('source')?.getAttribute('src') || videoEl.getAttribute('src');
      if (!src) return;

      // Normalize to a video_path accepted by the backend, relative to static (no leading /static/)
      const rel = src.replace(/^\/?static\//, '').replace(/^\//, '').replace(/^static\//, '');

      const resp = await fetch('/analyze_video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: rel })
      });

      if (!resp.ok) {
        console.warn('analyze_video error', resp.statusText);
        return;
      }

      const data = await resp.json();
      const feedBox = videoEl.closest('.feed-box');
      if (!feedBox) return;

      const statusEl = feedBox.querySelector('.status');
      // Use priority: speech -> sound for message
      if (data.final_threat) {
        statusEl.textContent = 'THREAT DETECTED';
        statusEl.className = 'status danger';
      } else if (data.speech_threat || data.sound_threat) {
        statusEl.textContent = 'POTENTIAL THREAT';
        statusEl.className = 'status danger';
      } else {
        statusEl.textContent = 'MONITORING';
        statusEl.className = 'status safe';
      }

      // Add or update a small transcription overlay inside the feed-box
      let transEl = feedBox.querySelector('.transcription');
      if (!transEl) {
        transEl = document.createElement('div');
        transEl.className = 'transcription';
        transEl.style.cssText = 'position:absolute;left:8px;bottom:28px;background:rgba(0,0,0,0.6);color:#fff;padding:6px 8px;border-radius:4px;font-size:12px;max-width:90%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
        feedBox.style.position = 'relative';
        feedBox.appendChild(transEl);
      }
      transEl.textContent = data.transcription ? data.transcription : (data.message || '');

    } catch (err) {
      console.error('analyzeFeed error', err);
    }
  }

  // Run initial analysis and then poll periodically
  function analyzeAllFeeds() {
    feeds.forEach(v => analyzeFeed(v));
  }

  analyzeAllFeeds();
  // Poll every 20 seconds (adjust as needed)
  setInterval(analyzeAllFeeds, 20000);
});