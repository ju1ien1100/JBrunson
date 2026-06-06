/**
 * Webgenta — MRT2 browser streaming client.
 *
 * Connects to the Python WebSocket server, sends a text prompt,
 * streams PCM audio through an AudioWorkletNode, and forwards
 * Web MIDI input as conditioning events.
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

type Status = "idle" | "connecting" | "streaming" | "error";

function setStatus(s: Status, detail = "") {
  statusEl.dataset.status = s;
  const labels: Record<Status, string> = {
    idle: "Idle",
    connecting: "Connecting…",
    streaming: "Streaming",
    error: `Error${detail ? ": " + detail : ""}`,
  };
  statusEl.textContent = labels[s];
  connectBtn.textContent = s === "streaming" ? "Disconnect" : "Connect";
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
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const [status, pitch, velocity] = e.data;
  const type = status & 0xf0;
  if (type === 0x90 && velocity > 0) {
    // Distinguish drums: MIDI channel 10 (status & 0x0f === 9)
    if ((status & 0x0f) === 9) {
      ws.send(JSON.stringify({ type: "drum", velocity }));
    } else {
      ws.send(JSON.stringify({ type: "noteon", pitch, velocity }));
    }
  } else if (type === 0x80 || (type === 0x90 && velocity === 0)) {
    if ((status & 0x0f) !== 9) {
      ws.send(JSON.stringify({ type: "noteoff", pitch }));
    }
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
    ws!.send(prompt);
    setStatus("streaming");
    startViz();
  };

  ws.onmessage = (e) => {
    if (workletNode && e.data instanceof ArrayBuffer) {
      workletNode.port.postMessage(new Float32Array(e.data));
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

// Allow updating the prompt mid-session
promptInput.addEventListener("change", () => {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "prompt", text: promptInput.value.trim() }));
  }
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
