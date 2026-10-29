// Empty string = same server (used in Colab or when the backend serves the frontend).
// Opened from a separate static server (port 5500) or a file, it talks to the local backend.
// For a separate deployment, put your backend URL here, e.g. "https://my-api.example.com".
const API_URL = (location.protocol === "file:" || location.port === "5500")
  ? "http://127.0.0.1:8000"
  : "";

const $ = (id) => document.getElementById(id);
const fileInput = $("fileInput"), dropzone = $("dropzone"), runBtn = $("runBtn");
const conf = $("conf"), confValue = $("confValue");
let mode = "image";
let selectedFile = null;

const CONFIG = {
  image: { accept: "image/*", text: "Drag and drop an image here, or click to browse" },
  video: { accept: "video/*", text: "Drag and drop a video here, or click to browse" },
};

function setMode(m) {
  mode = m;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === m));
  fileInput.accept = CONFIG[m].accept;
  $("dropText").textContent = CONFIG[m].text;
  selectedFile = null;
  fileInput.value = "";
  $("fileName").textContent = "";
  runBtn.disabled = true;
  hideError();
  $("results").hidden = true;
}

function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = false; }
function hideError() { $("error").hidden = true; }

function chooseFile(file) {
  if (!file) return;
  if (!file.type.startsWith(mode + "/")) {
    showError(`Please choose a ${mode} file.`);
    return;
  }
  hideError();
  selectedFile = file;
  $("fileName").textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  runBtn.disabled = false;
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
conf.addEventListener("input", () => (confValue.textContent = Number(conf.value).toFixed(2)));
confValue.textContent = Number(conf.value).toFixed(2);

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => chooseFile(e.dataTransfer.files[0]));

function statBox(value, label) {
  const d = document.createElement("div");
  d.className = "stat";
  const b = document.createElement("b"); b.textContent = value;
  const s = document.createElement("span"); s.textContent = label;
  d.append(b, s);
  return d;
}

const pct = (v) => (v * 100).toFixed(1) + "%";

function renderTable(detections) {
  const wrap = $("tableWrap");
  wrap.replaceChildren();
  if (!detections.length) return;
  const withId = detections.some((d) => d.person !== null && d.person !== undefined);
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  const cols = withId
    ? ["#", "Person", "Match", "Detection", "Closest matches"]
    : ["#", "Class", "Confidence", "Box (x1, y1, x2, y2)"];
  cols.forEach((h) => {
    const th = document.createElement("th"); th.textContent = h; head.appendChild(th);
  });
  const body = table.createTBody();
  detections.forEach((d, i) => {
    const row = body.insertRow();
    let cells;
    if (withId) {
      const others = d.person === "Unknown" ? d.candidates : d.candidates.slice(1);
      cells = [
        i + 1,
        d.person ?? "-",
        d.person_confidence != null ? pct(d.person_confidence) : "-",
        pct(d.confidence),
        others.map((c) => `${c.name} ${pct(c.confidence)}`).join(", ") || "-",
      ];
    } else {
      cells = [i + 1, d.class_name, pct(d.confidence), d.bbox.join(", ")];
    }
    cells.forEach((v) => { row.insertCell().textContent = v; });
  });
  wrap.appendChild(table);
}

function showImageResult(data) {
  const boxes = [statBox(data.count, "Signatures found")];
  if (data.identification_enabled) {
    const names = [...new Set(data.detections.map((d) => d.person).filter(Boolean))];
    boxes.push(statBox(names.join(", ") || "-", "Identified as"));
  }
  boxes.push(statBox(data.inference_ms + " ms", "Inference time"),
             statBox(`${data.width}x${data.height}`, "Image size"));
  $("stats").replaceChildren(...boxes);
  const img = document.createElement("img");
  img.src = "data:image/jpeg;base64," + data.image_base64;
  img.alt = "Detection result";
  $("media").replaceChildren(img);
  renderTable(data.detections);
}

function showVideoResult(data) {
  const stats = $("stats"); stats.replaceChildren(
    statBox(data.total_detections, "Total detections"),
    statBox(data.frames, "Frames processed"),
    statBox(data.processing_fps + " FPS", "Processing speed"));
  Object.entries(data.class_counts).forEach(([k, v]) => stats.appendChild(statBox(v, k)));
  const video = document.createElement("video");
  video.controls = true;
  video.src = API_URL + data.video_url;
  $("media").replaceChildren(video);
  const note = [];
  if (!data.browser_playable) note.push("Install ffmpeg on the server so every browser can play this video.");
  if (data.truncated) note.push("Only the first part of the video was processed.");
  const wrap = $("tableWrap"); wrap.replaceChildren();
  if (note.length) { const p = document.createElement("p"); p.textContent = note.join(" "); wrap.appendChild(p); }
}

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  $("results").hidden = true;
  $("loader").hidden = false;
  runBtn.disabled = true;

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("conf", conf.value);

  try {
    const res = await fetch(`${API_URL}/predict/${mode}`, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
    mode === "image" ? showImageResult(data) : showVideoResult(data);
    $("results").hidden = false;
    $("results").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError(err.message === "Failed to fetch"
      ? "Cannot reach the backend. Is the server running?" : err.message);
  } finally {
    $("loader").hidden = true;
    runBtn.disabled = false;
  }
});

async function checkServer() {
  const s = $("status");
  try {
    const res = await fetch(`${API_URL}/health`);
    const data = await res.json();
    s.textContent = data.identification
      ? `Server online | identity: ${data.people.length} people`
      : "Server online | detection only";
    s.className = "status ok";
  } catch {
    s.textContent = "Server offline";
    s.className = "status bad";
  }
}

setMode("image");
checkServer();
