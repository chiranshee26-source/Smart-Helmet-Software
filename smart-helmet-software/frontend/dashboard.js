const connStatus = document.getElementById("conn-status");
const alertBanner = document.getElementById("alert-banner");
const alertLevelEl = document.getElementById("alert-level");
const alertActionEl = document.getElementById("alert-action");
const cameraNote = document.getElementById("camera-note");

const earValue = document.getElementById("ear-value");
const perclosValue = document.getElementById("perclos-value");
const eyeScoreValue = document.getElementById("eye-score-value");

const pitchValue = document.getElementById("pitch-value");
const nodCountValue = document.getElementById("nod-count-value");
const headScoreValue = document.getElementById("head-score-value");

const scoreBarFill = document.getElementById("score-bar-fill");
const scoreValue = document.getElementById("score-value");
const eventLog = document.getElementById("event-log");

const earChart = new MiniChart(document.getElementById("ear-chart"), { min: 0, max: 0.5, lineColor: "#4da3ff" });
const pitchChart = new MiniChart(document.getElementById("pitch-chart"), { min: -40, max: 40, lineColor: "#ffb454" });

function MiniChart(canvas, opts) {
  const ctx = canvas.getContext("2d");
  const points = [];
  const maxPoints = 150;

  function resize() {
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio || 90 * devicePixelRatio;
  }
  resize();
  window.addEventListener("resize", resize);

  return {
    push(value) {
      if (value === null || value === undefined) return;
      points.push(value);
      if (points.length > maxPoints) points.shift();
      this.draw();
    },
    draw() {
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      if (points.length < 2) return;
      const range = opts.max - opts.min;
      ctx.beginPath();
      ctx.strokeStyle = opts.lineColor;
      ctx.lineWidth = 2 * devicePixelRatio;
      points.forEach((v, i) => {
        const x = (i / (maxPoints - 1)) * w;
        const norm = (v - opts.min) / range;
        const y = h - norm * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    },
  };
}

function setAlertClass(level) {
  alertBanner.classList.remove("alert-green", "alert-yellow", "alert-red");
  alertBanner.classList.add("alert-" + level.toLowerCase());
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    connStatus.textContent = "live";
    connStatus.className = "pill pill-online";
  };
  ws.onclose = () => {
    connStatus.textContent = "disconnected — retrying…";
    connStatus.className = "pill pill-offline";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();

  ws.onmessage = (msg) => {
    const s = JSON.parse(msg.data);

    alertLevelEl.textContent = s.alert_level;
    alertActionEl.textContent = s.action;
    setAlertClass(s.alert_level);

    cameraNote.textContent = s.camera_enabled ? "" : "(no camera detected — head channel only)";
    earValue.textContent = s.ear !== null ? s.ear.toFixed(3) : "—";
    perclosValue.textContent = (s.perclos * 100).toFixed(0) + "%";
    eyeScoreValue.textContent = s.eye_score.toFixed(2);
    earChart.push(s.ear);

    pitchValue.textContent = s.head_pitch_deg !== null ? s.head_pitch_deg.toFixed(1) + "°" : "—";
    nodCountValue.textContent = s.nod_count;
    headScoreValue.textContent = s.head_score.toFixed(2);
    pitchChart.push(s.head_pitch_deg);

    scoreBarFill.style.left = Math.min(100, s.composite_score * 100) + "%";
    scoreValue.textContent = s.composite_score.toFixed(2);

    if (s.events && s.events.length) {
      eventLog.innerHTML = "";
      s.events.slice().reverse().forEach(([ts, message]) => {
        const li = document.createElement("li");
        const time = new Date(ts * 1000).toLocaleTimeString();
        li.textContent = `[${time}] ${message}`;
        eventLog.appendChild(li);
      });
    }
  };
}
connect();

document.getElementById("btn-nod").addEventListener("click", () => {
  fetch("/demo/trigger_nod", { method: "POST" });
});
document.getElementById("btn-reset").addEventListener("click", () => {
  fetch("/demo/reset", { method: "POST" });
});
