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
    # Single container so session state (self._sessions) is always reachable.
    # Multiple containers would load-balance across instances, losing session dicts.
    min_containers=1,   # keep warm — eliminates cold-start delay entirely
    max_containers=1,   # all calls route to the same container
    scaledown_window=300,
)
class MagentaInference:
    # Preset names from presets.json — pre-embedded at container startup so
    # begin_session() for any preset is a dict lookup, not a TFLite round-trip.
    _PRESETS = [
        "jazz piano trio", "ambient pads", "cinematic strings", "drum and bass",
        "lo-fi hip hop", "synthwave", "funk groove", "classical piano",
        "reggae", "metal", "bossa nova", "bluegrass",
    ]

    @modal.enter()
    def load_model(self):
        import os
        os.environ["MAGENTA_HOME"] = VOLUME_PATH

        from magenta_rt.jax.system import MagentaRT2System
        print("Loading MRT2-small on GPU...", flush=True)
        self.system = MagentaRT2System(size="mrt2_small")
        self._sessions: dict = {}

        # Warm the TFLite text encoder kernels and cache all preset embeddings.
        # This adds ~5-10s to cold-start but makes every subsequent begin_session
        # for a known preset effectively free.
        print("Pre-embedding presets...", flush=True)
        self._style_cache: dict = {}
        for name in self._PRESETS:
            self._style_cache[name] = self.system.embed_style(name)
            print(f"  cached: {name}", flush=True)
        print("Model ready.", flush=True)

    # ── Session management ────────────────────────────────────────────────────

    def _embed(self, prompt: str):
        """Return cached embedding or compute and cache a new one."""
        if prompt not in self._style_cache:
            print(f"[modal] embedding new prompt: {prompt!r}", flush=True)
            self._style_cache[prompt] = self.system.embed_style(prompt)
        return self._style_cache[prompt]

    @modal.method()
    def begin_session(
        self,
        session_id: str,
        prompt: str,
        cfg_musiccoca: float = 5.0,
        cfg_notes: float = 3.0,
        cfg_drums: float = 1.0,
    ) -> None:
        """Initialise session state on the container (embedding served from cache)."""
        style = self._embed(prompt)
        self._sessions[session_id] = {
            "style": style,
            "state": None,
            "cfg_musiccoca": cfg_musiccoca,
            "cfg_notes": cfg_notes,
            "cfg_drums": cfg_drums,
        }
        print(f"[modal] session {session_id[:8]} started: {prompt!r}", flush=True)

    @modal.method()
    def update_session(
        self,
        session_id: str,
        prompt: str | None = None,
        cfg_musiccoca: float | None = None,
        cfg_notes: float | None = None,
        cfg_drums: float | None = None,
    ) -> None:
        """Update style prompt or CFG params for an active session."""
        s = self._sessions.get(session_id)
        if s is None:
            return
        if prompt is not None:
            s["style"] = self._embed(prompt)
            s["state"] = None  # reset generative state on style change
        if cfg_musiccoca is not None:
            s["cfg_musiccoca"] = cfg_musiccoca
        if cfg_notes is not None:
            s["cfg_notes"] = cfg_notes
        if cfg_drums is not None:
            s["cfg_drums"] = cfg_drums

    @modal.method()
    def end_session(self, session_id: str) -> None:
        """Free session state."""
        self._sessions.pop(session_id, None)

    # ── Rendering ─────────────────────────────────────────────────────────────

    @modal.method()
    def render_chunk(
        self,
        session_id: str,
        notes: list,
        frames: int = 25,
        strum: bool = False,
    ) -> bytes:
        """Render one chunk with live note conditioning.

        Onset is applied on frame 0; sustain (or re-onset if strum=True) on the
        remaining frames.  State is kept on the container — nothing is
        serialised over the network.

        Returns raw float32 PCM bytes, stereo interleaved, 48 kHz.
        """
        import numpy as np

        s = self._sessions.get(session_id)
        if s is None:
            raise ValueError(f"Unknown session {session_id!r} — call begin_session first")

        style = s["style"]
        state = s["state"]
        cfg_kw = dict(
            cfg_musiccoca=s["cfg_musiccoca"],
            cfg_notes=s["cfg_notes"],
            cfg_drums=s["cfg_drums"],
        )

        chunks = []

        # Frame 0: onset
        wf, state = self.system.generate(
            style=style, notes=notes, drums=[-1], frames=1, state=state, **cfg_kw,
        )
        chunks.append(wf.samples)

        # Frames 1+: sustain or continued onset (strum)
        if frames > 1:
            sustained = notes if strum else [min(v, 1) for v in notes]
            wf, state = self.system.generate(
                style=style, notes=sustained, drums=[-1], frames=frames - 1,
                state=state, **cfg_kw,
            )
            chunks.append(wf.samples)

        s["state"] = state
        return np.concatenate(chunks, axis=0).astype(np.float32).tobytes()

    @modal.method()
    def render_melody(self, session_id: str, segments: list) -> bytes:
        """Render a full melody sequence in one Modal call.

        Accepts the segment list produced by midi_library.build_segments()
        (each element: {"notes": [128 ints], "drums": [...], "frames": int}).
        Calls render_chunk locally per segment — no network RTT between segments.

        Returns concatenated float32 PCM bytes, stereo interleaved, 48 kHz.
        """
        import numpy as np

        pcm_chunks = []
        for seg in segments:
            chunk = self.render_chunk.local(session_id, seg["notes"], seg.get("frames", 1))
            pcm_chunks.append(np.frombuffer(chunk, dtype=np.float32))
        if not pcm_chunks:
            return b""
        return np.concatenate(pcm_chunks).astype(np.float32).tobytes()

    # ── Legacy helpers (kept for smoke_test) ─────────────────────────────────

    @modal.method()
    def embed_style(self, prompt: str) -> bytes:
        import numpy as np
        return np.array(self._embed(prompt), dtype=np.float32).tobytes()


@app.local_entrypoint()
def smoke_test():
    """Quick test: embed a style prompt and print the embedding shape."""
    inf = MagentaInference()
    style_bytes = inf.embed_style.remote("jazz piano")
    import numpy as np
    style = np.frombuffer(style_bytes, dtype=np.float32)
    print(f"embed_style OK — shape: {style.shape}, dtype: {style.dtype}")
