"""Modal deployment for MRT2 GPU inference.

Runs MagentaRT2System on a T4 GPU in the cloud.
Model weights are stored in a persistent Modal Volume so cold starts skip the download.

Setup (run once):
  pip install modal
  python -m modal setup
  modal run modal_magenta.py::download_checkpoints
  modal deploy modal_magenta.py

Then start the local server pointing at Modal:
  python ws_server.py --modal
"""

import modal
from pathlib import Path

MRT2_LOCAL = str(Path(__file__).parent.parent / "magenta-realtime")

# ── Persistent volume for model weights (~900 MB, downloaded once) ─────────────
volume = modal.Volume.from_name("magenta-weights", create_if_missing=True)
VOLUME_PATH = "/weights"
CHECKPOINT_DIR = f"{VOLUME_PATH}/magenta-rt-v2"

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Copy local magenta-realtime repo into the image — avoids any GitHub network issues
    .add_local_dir(MRT2_LOCAL, remote_path="/opt/magenta-realtime", copy=True)
    .run_commands(
        "pip install -e '/opt/magenta-realtime[jax]'",
        "pip install -e /opt/magenta-realtime/magenta_rt/_vendor/sequence-layers",
        "pip install huggingface_hub",
        # Upgrade to CUDA-enabled JAX (T4 = CUDA 12)
        "pip install -U 'jax[cuda12]'",
    )
)

app = modal.App("webgenta-magenta", image=image)

# ── One-time checkpoint download ───────────────────────────────────────────────

@app.function(volumes={VOLUME_PATH: volume}, timeout=600)
def download_checkpoints():
    """Download MRT2 weights from HuggingFace into the Modal volume. Run once."""
    import shutil
    import pathlib
    from huggingface_hub import hf_hub_download

    repo = "google/magenta-realtime-2"
    base = pathlib.Path(CHECKPOINT_DIR)

    files = [
        "checkpoints/mrt2_small.safetensors",
        "resources/musiccoca/audio_preprocessor.tflite",
        "resources/musiccoca/mapper.tflite",
        "resources/musiccoca/music_encoder.tflite",
        "resources/musiccoca/pretrained_vector_quantizer.tflite",
        "resources/musiccoca/spm.model",
        "resources/musiccoca/text_encoder.tflite",
        "resources/spectrostream/decoder.safetensors",
        "resources/spectrostream/encoder.safetensors",
        "resources/spectrostream/quantizer.safetensors",
    ]

    for rel_path in files:
        dst = base / rel_path
        if dst.exists():
            print(f"  already present: {rel_path}")
            continue
        print(f"  downloading {rel_path} ...")
        src = hf_hub_download(repo, rel_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        print(f"  -> {dst}")

    volume.commit()
    print("Done. Checkpoints saved to Modal volume.")


# ── Inference function ─────────────────────────────────────────────────────────

@app.cls(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=600,
    # Keep one container warm between requests to avoid cold-start delay
    min_containers=0,
    scaledown_window=300,  # stay warm for 5 min after last request
)
class MagentaInference:
    @modal.enter()
    def load_model(self):
        import os
        # MAGENTA_HOME is the parent of magenta-rt-v2/ — paths.py appends that subdir automatically
        os.environ["MAGENTA_HOME"] = VOLUME_PATH

        from magenta_rt.jax.system import MagentaRT2System
        print("Loading MRT2-small on GPU...", flush=True)
        self.system = MagentaRT2System(size="mrt2_small")
        print("Model ready.", flush=True)

    @modal.method()
    def embed_style(self, prompt: str) -> bytes:
        """Embed a text prompt → style vector bytes (float32)."""
        import numpy as np
        style = self.system.embed_style(prompt)
        return np.array(style, dtype=np.float32).tobytes()

    @modal.method()
    def render(self, style_bytes: bytes, notes_sequence: list) -> bytes:
        """Render a sequence of note segments → raw float32 PCM bytes (stereo interleaved).

        notes_sequence: list of {"notes": [128 ints], "drums": [int], "frames": int}
        Returns concatenated float32 PCM, stereo interleaved, 48kHz.
        """
        import numpy as np

        style = np.frombuffer(style_bytes, dtype=np.float32)
        state = None
        chunks = []

        for seg in notes_sequence:
            wf, state = self.system.generate(
                style=style,
                notes=seg["notes"],
                drums=seg["drums"],
                frames=seg["frames"],
                state=state,
            )
            chunks.append(wf.samples)

        audio = np.concatenate(chunks, axis=0).astype(np.float32)
        return audio.tobytes()


@app.local_entrypoint()
def smoke_test():
    """Quick test: embed a style prompt and print the embedding shape."""
    inf = MagentaInference()
    style_bytes = inf.embed_style.remote("jazz piano")
    import numpy as np
    style = np.frombuffer(style_bytes, dtype=np.float32)
    print(f"embed_style OK — shape: {style.shape}, dtype: {style.dtype}")
