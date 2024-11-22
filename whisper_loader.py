import torch
import torchaudio
from transformers import WhisperProcessor
import os

input_dir = "data/audio"
output_dir = "data/whisper_tiny"

# Load Whisper processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")


# Function to load and preprocess audio file
def preprocess_audio(file_path):
  try:
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)
    print(waveform.shape, sample_rate)

    # Resample to Whisper's expected sample rate if necessary
    if sample_rate != processor.feature_extractor.sampling_rate:
      resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=processor.feature_extractor.sampling_rate)
      waveform = resampler(waveform)

    print(waveform.shape, sample_rate)

    # Ensure mono channel
    if waveform.shape[0] > 1:
      waveform = torch.mean(waveform, dim=0, keepdim=True)

    print(waveform.shape, sample_rate)

    waveform = waveform.unsqueeze(0)

    print(waveform.shape, sample_rate)

    # Preprocess the audio to the model's input format
    inputs = processor(
        waveform,
        return_tensors="pt",
        sampling_rate=processor.feature_extractor.sampling_rate,
        padding=True,
        truncation=True
    )

    return inputs.input_features
  except Exception as e:
    # Handle loading errors here (e.g., print message, skip file)
    print(f"Error loading file {file_path}: {e}")
    pass  # You can replace 'pass' with specific actions for handling errors


# Function to save tensor to file
def save_tensor(tensor, file_path):
  torch.save(tensor, file_path)

def main():
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
      if file_name.endswith(".mp3"):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name.replace(".mp3", ".pt"))

      try:
        tensor = preprocess_audio(input_file_path)
        tensor = torch.squeeze(tensor, dim=0)
        save_tensor(tensor, output_file_path)
        print(f"Processed and saved: {file_name}")
      except Exception as e:
        # Handle errors during processing or saving (e.g., print message)
        print(f"Error processing {file_name}: {e}")
  else:
    print('Whisper Spectrograms already created!')