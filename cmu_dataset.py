import os
import re
import torch
import torchaudio
from torch.utils.data import Dataset


class CMUArcticDataset(Dataset):
    """
    Custom Dataset for CMU Arctic bdl speaker.
    Loads paired audio (WAV) and text transcriptions.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.wav_dir = os.path.join(root_dir, "wav")
        self.trans_file = os.path.join(root_dir, "etc", "txt.done.data")

        # Load transcriptions
        self.transcriptions = {}
        with open(self.trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                match = re.match(r'\( (\S+) "(.*)" \)', line)
                if match:
                    clip_id, text = match.groups()
                    self.transcriptions[clip_id] = text

        # Load WAV files
        self.wav_files = [f for f in os.listdir(self.wav_dir) if f.endswith(".wav")]
        self.wav_ids = [os.path.splitext(f)[0] for f in self.wav_files]

        # Ensure paired data only
        self.paired_ids = [wid for wid in self.wav_ids if wid in self.transcriptions]
        self.wav_files = [f"{wid}.wav" for wid in self.paired_ids]

        # Verify counts
        print(f"Loaded {len(self.paired_ids)} paired audio-text samples")
        if len(self.paired_ids) != len(self.transcriptions):
            print(f"Warning: {len(self.transcriptions)} texts, {len(self.wav_files)} WAVs")

    def __len__(self):
        return len(self.paired_ids)

    def __getitem__(self, idx):
        """
        Returns: waveform (tensor), text (str), clip_id (str)
        """
        clip_id = self.paired_ids[idx]
        wav_path = os.path.join(self.wav_dir, f"{clip_id}.wav")
        text = self.transcriptions[clip_id]

        # Load audio
        waveform, sample_rate = torchaudio.load(wav_path)
        assert sample_rate == 16000, f"Expected 16kHz, got {sample_rate}"

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, text, clip_id
