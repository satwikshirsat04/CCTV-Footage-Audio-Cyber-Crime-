document.addEventListener('DOMContentLoaded', () => {

  // ===============================
  // ELEMENTS
  // ===============================
  const feeds = Array.from(document.querySelectorAll('video[id^="feed"]'));
  const globalStatusText = document.getElementById('global-status-text');
  const globalStatusMsg = document.getElementById('global-status-msg');
  const themeToggle = document.getElementById('theme-toggle');

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

    if (!src) return false;

    const videoPath = src.replace(/^\/?static\//, '');

    try {
      const res = await fetch('/analyze_video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoPath })
      });

      if (!res.ok) return false;

      const data = await res.json();
      const feedBox = videoEl.closest('.feed-box');
      const statusEl = feedBox.querySelector('.status');

      // ===============================
      // CAMERA STATUS + BLINK
      // ===============================
      if (data.final_threat) {
        statusEl.textContent = 'THREAT DETECTED';
        statusEl.className = 'status danger';

        // 🔔 blinking red border
        feedBox.classList.add('threat');

        return true;
      } else {
        statusEl.textContent = 'MONITORING';
        statusEl.className = 'status safe';

        // remove blink
        feedBox.classList.remove('threat');
      }

      // ===============================
      // TRANSCRIPTION OVERLAY
      // ===============================
      let overlay = feedBox.querySelector('.transcription');
      if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'transcription';
        overlay.style.cssText =
          'position:absolute;bottom:30px;left:10px;background:rgba(0,0,0,0.65);color:#fff;padding:6px 10px;border-radius:5px;font-size:12px;max-width:90%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;';
        feedBox.style.position = 'relative';
        feedBox.appendChild(overlay);
      }

      overlay.textContent = data.transcription || '';

      return false;

    } catch (err) {
      console.error('Analysis error:', err);
      return false;
    }
  }

  // ===============================
  // ANALYZE ALL FEEDS
  // ===============================
  async function analyzeAllFeeds() {
    let anyThreat = false;

    for (const video of feeds) {
      const threat = await analyzeFeed(video);
      if (threat) anyThreat = true;
    }

    // ===============================
    // GLOBAL STATUS
    // ===============================
    if (anyThreat) {
      globalStatusText.textContent = 'ALERT';
      globalStatusText.className = 'status danger';
      globalStatusMsg.textContent = 'Crime detected in one or more cameras';
    } else {
      globalStatusText.textContent = 'MONITORING';
      globalStatusText.className = 'status safe';
      globalStatusMsg.textContent = 'All systems operational';
    }
  }

  // ===============================
  // INIT
  // ===============================
  updateTime();
  analyzeAllFeeds();

  setInterval(updateTime, 60000);        // clock
  setInterval(analyzeAllFeeds, 20000);  // AI polling
});
