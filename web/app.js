const form = document.querySelector("#generate-form");
const audioInput = document.querySelector("#audio");
const fileLabel = document.querySelector("#file-label");
const statusEl = document.querySelector("#status");
const temperature = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperature-value");
const metricsEl = document.querySelector("#metrics");
const tokensEl = document.querySelector("#tokens");
const objectsEl = document.querySelector("#objects");
const canvas = document.querySelector("#preview");
const ctx = canvas.getContext("2d");
const exportJson = document.querySelector("#export-json");
const exportObjects = document.querySelector("#export-objects");
const exportLevel = document.querySelector("#export-level");
const exportGmd = document.querySelector("#export-gmd");
const saveFile = document.querySelector("#save-file");
const saveLabel = document.querySelector("#save-label");
const injectSave = document.querySelector("#inject-save");
const targetLevelKey = document.querySelector("#target-level-key");

let currentGeneration = null;

audioInput.addEventListener("change", () => {
  const file = audioInput.files[0];
  fileLabel.textContent = file ? file.name : "Song Upload";
});

temperature.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = audioInput.files[0];
  if (!file) {
    setStatus("Select audio", "error");
    return;
  }

  const body = new FormData();
  body.append("audio", file);
  body.append("difficulty", document.querySelector("#difficulty").value);
  body.append("temperature", temperature.value);
  body.append("top_k", document.querySelector("#top-k").value);
  body.append("max_tokens", document.querySelector("#max-tokens").value);
  body.append("seed", document.querySelector("#seed").value);
  body.append("alignments", selectedAlignments().join(","));

  setBusy(true);
  setStatus("Generating", "busy");
  try {
    const response = await fetch("/generate", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "generation_failed");
    }
    currentGeneration = payload;
    renderGeneration(payload);
    setStatus("Ready");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

exportJson.addEventListener("click", () => openExport("json"));
exportObjects.addEventListener("click", () => openExport("object_strings"));
exportLevel.addEventListener("click", () => openExport("level_string"));
exportGmd.addEventListener("click", () => openExport("gmd"));
saveFile.addEventListener("change", () => {
  saveLabel.textContent = saveFile.files[0] ? saveFile.files[0].name : "CCGameManager.dat";
});
injectSave.addEventListener("click", injectCurrentSave);

window.addEventListener("resize", () => {
  if (currentGeneration) {
    drawPreview(currentGeneration);
  } else {
    drawEmpty();
  }
});

drawEmpty();

function selectedAlignments() {
  return [...document.querySelectorAll("input[name='alignment']:checked")].map((item) => item.value);
}

function setBusy(value) {
  document.querySelector("#generate-button").disabled = value;
}

function setStatus(text, state = "") {
  statusEl.textContent = text;
  statusEl.className = `status ${state}`.trim();
}

function renderGeneration(payload) {
  const metrics = payload.metrics;
  const audio = payload.audio;
  metricsEl.innerHTML = "";
  [
    `BPM ${audio.bpm}`,
    `${metrics.sequence_length} tokens`,
    `${metrics.object_count} objects`,
    `${metrics.portal_count} portals`,
    `${metrics.orb_count} orbs`,
    `${metrics.tokens_per_second}/s`
  ].forEach((text) => {
    const item = document.createElement("span");
    item.className = "metric";
    item.textContent = text;
    metricsEl.appendChild(item);
  });

  tokensEl.textContent = payload.tokens.join(" ");
  objectsEl.textContent = JSON.stringify(payload.preview.objects.slice(0, 160), null, 2);
  exportJson.disabled = false;
  exportObjects.disabled = false;
  exportLevel.disabled = false;
  exportGmd.disabled = false;
  injectSave.disabled = false;
  drawPreview(payload);
}

function drawEmpty() {
  resizeCanvas();
  ctx.fillStyle = "#11161b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawGrid(0, 40, 16, 1);
}

function drawPreview(payload) {
  resizeCanvas();
  const objects = payload.preview.objects;
  const widthSteps = Math.max(payload.preview.width_steps, 30);
  const lanes = Math.max(payload.preview.height_lanes, 16);
  const scaleX = canvas.width / Math.max(widthSteps * 30 + 160, 1);
  const laneHeight = canvas.height / (lanes + 3);
  const scale = Math.max(0.35, Math.min(scaleX, laneHeight / 30));
  const xOffset = 36;
  const yBase = canvas.height - 48;

  ctx.fillStyle = "#11161b";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawEnergy(payload, yBase, scale, xOffset);
  drawGrid(yBase, widthSteps, lanes, scale, xOffset);

  for (const object of objects) {
    const x = xOffset + object.x_step * 30 * scale;
    const y = yBase - object.y_lane * 30 * scale;
    drawObject(object, x, y, scale);
  }
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(640, Math.floor(rect.width * ratio));
  const height = Math.max(320, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function drawGrid(yBase = 0, steps = 40, lanes = 16, scale = 1, xOffset = 36) {
  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  for (let step = 0; step <= steps; step += 2) {
    const x = xOffset + step * 30 * scale;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
  }
  for (let lane = 0; lane <= lanes; lane += 2) {
    const y = yBase ? yBase - lane * 30 * scale : canvas.height - 40 - lane * 18;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#53606d";
  ctx.beginPath();
  ctx.moveTo(0, yBase || canvas.height - 40);
  ctx.lineTo(canvas.width, yBase || canvas.height - 40);
  ctx.stroke();
}

function drawEnergy(payload, yBase, scale, xOffset) {
  const energy = payload.audio.energy || [];
  const stepSeconds = payload.conditioning.step_seconds || 0.2;
  if (!energy.length) return;
  ctx.beginPath();
  for (const point of energy) {
    const step = point.time / stepSeconds;
    const x = xOffset + step * 30 * scale;
    const y = yBase - 24 - point.value * 90 * scale;
    if (point === energy[0]) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "rgba(72, 209, 143, 0.45)";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawObject(object, x, y, scale) {
  const size = Math.max(8, 24 * scale);
  if (object.token === "BLOCK" || object.token === "PLATFORM") {
    const width = Math.max(size, size * (object.width || 1));
    ctx.fillStyle = object.token === "BLOCK" ? "#5aa7ff" : "#48d18f";
    ctx.fillRect(x, y - size, width, size);
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.strokeRect(x, y - size, width, size);
    return;
  }

  if (object.token === "SPIKE") {
    ctx.fillStyle = "#f2ba55";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + size * 0.5, y - size);
    ctx.lineTo(x + size, y);
    ctx.closePath();
    ctx.fill();
    return;
  }

  if (object.token.startsWith("ORB")) {
    ctx.strokeStyle = object.token.includes("BLUE") ? "#5aa7ff" : object.token.includes("BLACK") ? "#d4dde5" : "#f2ba55";
    ctx.lineWidth = Math.max(2, 3 * scale);
    ctx.beginPath();
    ctx.arc(x + size * 0.5, y - size * 0.5, size * 0.38, 0, Math.PI * 2);
    ctx.stroke();
    return;
  }

  if (object.token.startsWith("PORTAL")) {
    ctx.strokeStyle = object.token.includes("WAVE") ? "#ff6f6a" : "#b089ff";
    ctx.lineWidth = Math.max(2, 3 * scale);
    ctx.beginPath();
    ctx.ellipse(x + size * 0.45, y - size * 0.6, size * 0.42, size * 0.75, 0, 0, Math.PI * 2);
    ctx.stroke();
    return;
  }

  ctx.fillStyle = object.token.startsWith("SPEED") ? "#ff6f6a" : "#d4dde5";
  ctx.fillRect(x, y - size, size, size);
}

function openExport(format) {
  if (!currentGeneration) return;
  window.open(`/export/${currentGeneration.id}?format=${format}`, "_blank", "noopener");
}

async function injectCurrentSave() {
  if (!currentGeneration || !saveFile.files[0]) {
    setStatus("Select save", "error");
    return;
  }
  const body = new FormData();
  body.append("save", saveFile.files[0]);
  body.append("target_level_key", targetLevelKey.value || "");

  setStatus("Injecting", "busy");
  injectSave.disabled = true;
  try {
    const response = await fetch(`/inject/${currentGeneration.id}`, { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "inject_failed");
    }
    objectsEl.textContent = JSON.stringify(payload, null, 2);
    setStatus("Save written");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    injectSave.disabled = false;
  }
}
