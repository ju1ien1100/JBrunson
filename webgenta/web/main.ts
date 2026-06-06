/**
 * Webgenta — MRT2 browser streaming client.
 *
 * Connects to ws_server.py, sends { service, action, ... } JSON messages,
 * receives JSON events + binary PCM chunks, and plays audio via AudioWorklet.
 */

// ─── State ──────────────────────────────────────────────────────────────────

let audioCtx: AudioContext | null = null;
let workletNode: AudioWorkletNode | null = null;
let analyser: AnalyserNode | null = null;
let ws: WebSocket | null = null;
let midiAccess: MIDIAccess | null = null;
let animFrame: number | null = null;

// ─── DOM refs ────────────────────────────────────────────────────────────────

const promptInput = document.getElementById("prompt") as HTMLInputElement;
const serverInput = document.getElementById("server") as HTMLInputElement;
const connectBtn = document.getElementById("connect-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status") as HTMLSpanElement;
const midiSelect = document.getElementById("midi-select") as HTMLSelectElement;
const canvas = document.getElementById("viz") as HTMLCanvasElement;
const ctx2d = canvas.getContext("2d")!;

// ─── Status helpers ──────────────────────────────────────────────────────────

type Status = "idle" | "connecting" | "embedding" | "streaming" | "error";

function setStatus(s: Status, detail = "") {
  statusEl.dataset.status = s;
  const labels: Record<Status, string> = {
    idle: "Idle",
    connecting: "Connecting…",
    embedding: "Embedding prompt…",
    streaming: "Streaming",
    error: `Error${detail ? ": " + detail : ""}`,
  };
  statusEl.textContent = labels[s];
  connectBtn.textContent = s === "streaming" || s === "embedding" ? "Disconnect" : "Connect";
}

// ─── Server message helpers ──────────────────────────────────────────────────

function send(action: string, payload: Record<string, unknown> = {}) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ service: "magenta", action, ...payload }));
  }
}

function handleServerMessage(data: unknown) {
  if (!(data instanceof ArrayBuffer)) return;
  workletNode?.port.postMessage(new Float32Array(data));
}

function handleServerEvent(msg: Record<string, unknown>) {
  const event = msg.event as string;
  if (event === "status") {
    const state = msg.state as string;
    if (state === "embedding" || state === "rendering") setStatus("embedding");
  } else if (event === "ready") {
    setStatus("streaming");
    const loop = msg.loop as number;
    const dur = msg.duration_s as number;
    console.log(`[magenta] loop ${loop} ready — ${dur.toFixed(1)}s of audio`);
  } else if (event === "stopped") {
    teardown();
  } else if (event === "error") {
    setStatus("error", msg.message as string);
    teardown();
  }
}

// ─── MIDI ────────────────────────────────────────────────────────────────────

async function initMidi() {
  if (!navigator.requestMIDIAccess) return;
  try {
    midiAccess = await navigator.requestMIDIAccess();
    populateMidiSelect();
    midiAccess.onstatechange = populateMidiSelect;
  } catch {
    console.warn("Web MIDI not available");
  }
}

function populateMidiSelect() {
  if (!midiAccess) return;
  const prev = midiSelect.value;
  midiSelect.innerHTML = '<option value="">— No MIDI —</option>';
  midiAccess.inputs.forEach((input) => {
    const opt = document.createElement("option");
    opt.value = input.id;
    opt.textContent = input.name ?? input.id;
    midiSelect.appendChild(opt);
  });
  if (prev) midiSelect.value = prev;
  bindMidiInput();
}

function bindMidiInput() {
  if (!midiAccess) return;
  midiAccess.inputs.forEach((input) => {
    input.onmidimessage = midiSelect.value === input.id ? onMidiMessage : null;
  });
}

midiSelect.addEventListener("change", bindMidiInput);

function onMidiMessage(e: MIDIMessageEvent) {
  const [status, pitch, velocity] = e.data;
  const type = status & 0xf0;
  const isDrum = (status & 0x0f) === 9;

  if (type === 0x90 && velocity > 0) {
    if (isDrum) {
      send("drum", { velocity });
    } else {
      send("note_on", { pitch, velocity });
    }
  } else if (type === 0x80 || (type === 0x90 && velocity === 0)) {
    if (!isDrum) send("note_off", { pitch });
  }
}

// ─── Audio setup ─────────────────────────────────────────────────────────────

async function initAudio(serverUrl: string, prompt: string) {
  audioCtx = new AudioContext({ sampleRate: 48000 });
  await audioCtx.audioWorklet.addModule("/mrt2-worklet.js");

  workletNode = new AudioWorkletNode(audioCtx, "mrt2-processor", {
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });

  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  workletNode.connect(analyser);
  analyser.connect(audioCtx.destination);

  ws = new WebSocket(serverUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    // New protocol: start the Magenta service with a prompt
    send("start", { prompt });
    setStatus("embedding");
    startViz();
  };

  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      handleServerMessage(e.data);
    } else {
      try {
        handleServerEvent(JSON.parse(e.data as string));
      } catch {
        // ignore malformed
      }
    }
  };

  ws.onerror = () => setStatus("error", "WebSocket error");
  ws.onclose = () => {
    setStatus("idle");
    stopViz();
  };
}

function teardown() {
  ws?.close();
  ws = null;
  workletNode?.disconnect();
  workletNode = null;
  audioCtx?.close();
  audioCtx = null;
  stopViz();
}

// ─── Connect / disconnect ─────────────────────────────────────────────────────

connectBtn.addEventListener("click", async () => {
  if (ws) {
    teardown();
    return;
  }

  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus("error", "Enter a prompt first");
    return;
  }

  setStatus("connecting");
  try {
    await initAudio(serverInput.value.trim(), prompt);
  } catch (err) {
    setStatus("error", String(err));
    teardown();
  }
});

// Update style prompt mid-session
promptInput.addEventListener("change", () => {
  send("prompt", { text: promptInput.value.trim() });
});

// ─── Visualizer ──────────────────────────────────────────────────────────────

function startViz() {
  const buf = new Float32Array(analyser!.fftSize);
  function draw() {
    animFrame = requestAnimationFrame(draw);
    analyser!.getFloatTimeDomainData(buf);
    const w = canvas.width, h = canvas.height;
    ctx2d.fillStyle = "#0a0a0a";
    ctx2d.fillRect(0, 0, w, h);
    ctx2d.strokeStyle = "#7c3aed";
    ctx2d.lineWidth = 1.5;
    ctx2d.beginPath();
    const step = w / buf.length;
    for (let i = 0; i < buf.length; i++) {
      const x = i * step;
      const y = (1 - (buf[i] + 1) / 2) * h;
      i === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y);
    }
    ctx2d.stroke();
  }
  draw();
}

function stopViz() {
  if (animFrame !== null) cancelAnimationFrame(animFrame);
  animFrame = null;
  ctx2d.fillStyle = "#0a0a0a";
  ctx2d.fillRect(0, 0, canvas.width, canvas.height);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

initMidi();
setStatus("idle");
