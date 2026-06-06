"""Modal deployment for Stable Audio 3 Small GPU inference.

Runs stabilityai/stable-audio-3-small-music on a T4 GPU.
Model weights are cached in a persistent Modal Volume.

Setup (run once):
  modal deploy webgenta/modal_stability.py

Then start the server with:
  python ws_server.py --modal --modal-stability
"""

import modal

volume = modal.Volume.from_name("stability-weights", create_if_missing=True)
VOLUME_PATH = "/weights"
MODEL_DIR = f"{VOLUME_PATH}/stable-audio-3-small"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .run_commands(
        "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install 'numpy<2.0'",
        "pip install pytorch-lightning git+https://github.com/Stability-AI/stable-audio-tools huggingface_hub einops",
        # stable-audio-tools pins protobuf==3.19.6 via wandb/tensorboard which
        # conflicts with Modal's runner. Force a compatible version last.
        "pip install 'protobuf>=4.25.0,<6'",
    )
)

app = modal.App("webgenta-stability", image=image)


@app.cls(
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=300,
    min_containers=0,
    scaledown_window=300,
)
class StableAudioInference:
    @modal.enter()
    def load_model(self):
        import os
        import torch
        from stable_audio_tools import get_pretrained_model

        os.environ["HF_HOME"] = MODEL_DIR
        print("Loading Stable Audio 3 Small...", flush=True)
        self.model, self.config = get_pretrained_model("stabilityai/stable-audio-3-small-music")
        self.sample_rate = self.config["sample_rate"]
        self.sample_size = self.config["sample_size"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).to(torch.float16)
        print(f"Model ready on {self.device}.", flush=True)

    @modal.method()
    def generate(self, prompt: str, duration: int = 30) -> bytes:
        """Generate audio from a text prompt. Returns WAV bytes (stereo, 44100 Hz)."""
        import io
        import torch
        from einops import rearrange
        from scipy.io import wavfile
        from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

        conditioning = [{"prompt": prompt, "seconds_total": duration}]
        output = generate_diffusion_cond_inpaint(
            self.model,
            steps=8,
            cfg_scale=1.0,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sampler_type="pingpong",
            device=self.device,
        )
        output = rearrange(output, "b d n -> d (b n)")
        arr = (output.to(torch.float32)
               .div(torch.max(torch.abs(output)))
               .clamp(-1, 1)
               .mul(32767)
               .to(torch.int16)
               .cpu()
               .numpy())  # (channels, samples)
        buf = io.BytesIO()
        wavfile.write(buf, self.sample_rate, arr.T)  # scipy wants (samples, channels)
        return buf.getvalue()


@app.local_entrypoint()
def smoke_test():
    """Generate a 5-second clip and print the WAV size."""
    inf = StableAudioInference()
    wav = inf.generate.remote("ambient lo-fi piano, peaceful", duration=5)
    print(f"generate OK — WAV size: {len(wav) / 1024:.1f} KB")
