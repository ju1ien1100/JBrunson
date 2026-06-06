/**
 * AudioWorklet processor for MRT2 streaming audio.
 * Served as a static file from public/ — Vite cannot bundle AudioWorklet modules.
 *
 * Frame format from server: 3840 floats = 1920 samples × 2ch @ 48kHz (40ms)
 * Interleaved layout: [L0, R0, L1, R1, ..., L1919, R1919]
 */

const RING_SAMPLES = 48000 * 2; // 2-second ring buffer per channel

class MRT2Processor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ringL = new Float32Array(RING_SAMPLES);
    this.ringR = new Float32Array(RING_SAMPLES);
    this.writeHead = 0;
    this.readHead = 0;

    this.port.onmessage = (e) => {
      const chunk = e.data; // interleaved [L, R, L, R, ...]
      const frames = chunk.length >> 1;
      for (let i = 0; i < frames; i++) {
        // Drop frames if buffer is full to prevent write head lapping read head
        if (this.writeHead - this.readHead >= RING_SAMPLES) break;
        this.ringL[this.writeHead % RING_SAMPLES] = chunk[i * 2];
        this.ringR[this.writeHead % RING_SAMPLES] = chunk[i * 2 + 1];
        this.writeHead++;
      }
    };
  }

  process(_inputs, outputs) {
    const outL = outputs[0]?.[0];
    const outR = outputs[0]?.[1];
    if (!outL) return true;

    const available = this.writeHead - this.readHead;
    const blockSize = outL.length;

    if (available < blockSize) {
      outL.fill(0);
      if (outR) outR.fill(0);
      return true;
    }

    for (let i = 0; i < blockSize; i++) {
      outL[i] = this.ringL[this.readHead % RING_SAMPLES];
      if (outR) outR[i] = this.ringR[this.readHead % RING_SAMPLES];
      this.readHead++;
    }
    return true;
  }
}

registerProcessor("mrt2-processor", MRT2Processor);
