import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchaudio.pipelines import TACOTRON2_WAVERNN_PHONE_LJSPEECH
from cmu_dataset import CMUArcticDataset
from dp.preprocessing.text import Preprocessor, LanguageTokenizer, SequenceTokenizer
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Custom collate function to pad waveforms
def collate_fn(batch):
    """
    Pads waveforms to the longest in the batch, keeps text and clip_id as lists.
    Args:
        batch: List of (waveform, text, clip_id) tuples
    Returns:
        (padded_waveforms, texts, clip_ids)
    """
    waveforms, texts, clip_ids = zip(*batch)

    # Skip invalid waveforms
    valid_waveforms = [w for w in waveforms if w.abs().sum() > 0]
    if not valid_waveforms:
        return None

    # Find max length
    max_len = max(w.shape[-1] for w in valid_waveforms)

    # Pad waveforms
    padded_waveforms = []
    for w in waveforms:
        if w.abs().sum() == 0:
            padded_waveforms.append(torch.zeros(1, max_len))
        else:
            pad_len = max_len - w.shape[-1]
            if pad_len > 0:
                w = torch.nn.functional.pad(w, (0, pad_len))
            padded_waveforms.append(w)

    return (
        torch.stack(padded_waveforms),
        list(texts),
        list(clip_ids)
    )


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "cmu_us_bdl_arctic")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cpu")
print(f"Using device: {device}")

# Load pre-trained Tacotron 2 bundle with safe globals
bundle = TACOTRON2_WAVERNN_PHONE_LJSPEECH
with torch.serialization.safe_globals([Preprocessor, LanguageTokenizer, SequenceTokenizer]):
    processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)
tacotron2.train()
vocoder.eval()

# Dataset
dataset = CMUArcticDataset(root_dir=DATA_DIR)

# Split dataset into 80% train and 20% test
train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # 20% for testing

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# DataLoader for train and test datasets
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    collate_fn=collate_fn
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    collate_fn=collate_fn
)

# Optimizer
optimizer = torch.optim.Adam(tacotron2.parameters(), lr=1e-4)

# Mel-spectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=80,
    n_fft=1024,
    hop_length=256,
    f_min=0,
    f_max=8000
).to(device)

# Lists to store loss values for plotting
train_losses = []
test_losses = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    total_train_loss = 0
    tacotron2.train()  # Ensure model is in training mode
    for batch_idx, batch in enumerate(train_dataloader):
        if batch is None:
            print(f"Skipping batch {batch_idx} due to invalid waveforms")
            continue

        waveform, text, clip_id = batch
        waveform = waveform.to(device)

        # Process text to phonemes
        try:
            processed, lengths = processor(text)
        except Exception as e:
            print(f"Text processing failed for batch {batch_idx}: {e}")
            continue
        processed = processed.to(device)
        lengths = lengths.to(device)

        # Ground-truth mel-spectrogram
        mel_gt = mel_transform(waveform.squeeze(1))

        # Compute mel-spectrogram lengths
        mel_lengths = torch.tensor([mel_gt.shape[-1]] * waveform.shape[0], dtype=torch.int32).to(device)

        # Sort by lengths in decreasing order
        lengths, indices = torch.sort(lengths, descending=True)
        processed = processed[indices]
        mel_gt = mel_gt[indices]
        mel_lengths = mel_lengths[indices]

        # Forward pass: Tacotron 2
        optimizer.zero_grad()

        outputs = tacotron2(processed, lengths, mel_gt, mel_lengths)

        mel_pred = outputs[0]  # Mel-spectrogram predictions
        loss = torch.nn.functional.l1_loss(mel_pred, mel_gt)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tacotron2.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Average Train Loss: {avg_train_loss:.4f}")
    train_losses.append(avg_train_loss)  # Store training loss for this epoch

    # Evaluation phase (on test set)
    total_test_loss = 0
    tacotron2.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch is None:
                print(f"Skipping batch {batch_idx} due to invalid waveforms")
                continue

            waveform, text, clip_id = batch
            waveform = waveform.to(device)

            # Process text to phonemes
            try:
                processed, lengths = processor(text)
            except Exception as e:
                print(f"Text processing failed for batch {batch_idx}: {e}")
                continue
            processed = processed.to(device)
            lengths = lengths.to(device)

            # Ground-truth mel-spectrogram
            mel_gt = mel_transform(waveform.squeeze(1))

            # Compute mel-spectrogram lengths
            mel_lengths = torch.tensor([mel_gt.shape[-1]] * waveform.shape[0], dtype=torch.int32).to(device)

            # Sort by lengths in decreasing order
            lengths, indices = torch.sort(lengths, descending=True)
            processed = processed[indices]
            mel_gt = mel_gt[indices]
            mel_lengths = mel_lengths[indices]

            # Forward pass: Tacotron 2 (no gradients needed in evaluation)
            outputs = tacotron2(processed, lengths, mel_gt, mel_lengths)

            mel_pred = outputs[0]  # Mel-spectrogram predictions
            loss = torch.nn.functional.l1_loss(mel_pred, mel_gt)

            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"Epoch {epoch + 1}, Average Test Loss: {avg_test_loss:.4f}")
    test_losses.append(avg_test_loss)  # Store test loss for this epoch

    # Save checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, f"tacotron2_epoch_{epoch + 1}.pth")
    torch.save(tacotron2.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

# After training, plot the learning curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
