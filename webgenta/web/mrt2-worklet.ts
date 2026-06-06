/**
 * AudioWorklet processor for MRT2 streaming audio.
 *
 * Receives stereo interleaved Float32 chunks from the main thread
 * (posted via port.postMessage) and plays them continuously.
 *
 * Frame format from server: 3840 floats = 1920 samples × 2ch @ 48kHz (40ms)
 * Interleaved layout: [L0, R0, L1, R1, ..., L1919, R1919]
 */

const RING_SAMPLES = 48000 * 4; // 4-second ring buffer per channel

class MRT2Processor extends AudioWorkletProcessor {
  private ringL = new Float32Array(RING_SAMPLES);
  private ringR = new Float32Array(RING_SAMPLES);
  private writeHead = 0;
  private readHead = 0;

  constructor() {
    super();
    this.port.onmessage = (e: MessageEvent<Float32Array>) => {
      const chunk = e.data; // interleaved [L, R, L, R, ...]
      const frames = chunk.length >> 1;
      for (let i = 0; i < frames; i++) {
        this.ringL[this.writeHead % RING_SAMPLES] = chunk[i * 2];
        this.ringR[this.writeHead % RING_SAMPLES] = chunk[i * 2 + 1];
        this.writeHead++;
      }
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const outL = outputs[0]?.[0];
    const outR = outputs[0]?.[1];
    if (!outL) return true;

    const available = this.writeHead - this.readHead;
    const blockSize = outL.length;

    if (available < blockSize) {
      // Buffer underrun — output silence rather than glitch
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
