import torch
import torchaudio
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "tacotron2_epoch_10.pth")
OUTPUT_WAV = os.path.join(BASE_DIR, "output.wav")

# Device
device = torch.device("cpu")
print(f"Using device: {device}")

# Load pre-trained bundle
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

# Load fine-tuned weights
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
tacotron2.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
tacotron2.eval()
vocoder.eval()

# Input text
text = "Hello, this is a test of text to speech with Tacotron two."

# Process text
processed, lengths = processor([text])
processed = processed.to(device)
lengths = lengths.to(device)

# Generate mel-spectrogram
with torch.no_grad():
    mel_pred, _, _ = tacotron2(processed, lengths)

# Convert to waveform
with torch.no_grad():
    waveform = vocoder(mel_pred).squeeze(0)

# Save output
torchaudio.save(OUTPUT_WAV, waveform.cpu(), sample_rate=16000)
print(f"Saved output to {OUTPUT_WAV}")
