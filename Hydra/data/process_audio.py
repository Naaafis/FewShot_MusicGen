import os
import torch
import torchaudio
import sys
from tqdm import tqdm

def process_directory(input_directory):
    """
    Convert all stereo files to mono in the provided directory and its subdirectories using torchaudio.
    If a file is empty and has no sound, then delete the file.
    """

    # Walk through input directory and its subdirectories
    all_files = [os.path.join(root, name) for root, dirs, files in os.walk(input_directory) for name in files]
    wav_files = [file for file in all_files if file.endswith(".wav")]
    for file_name in tqdm(wav_files, desc="Processing audio files", unit="file"):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_name)

        # Check if waveform is stereo
        if waveform.shape[0] > 1:
            # Convert to mono by averaging channels
            waveform = waveform.mean(dim=0, keepdim=True)

            # Save the mono audio back to the same file
            torchaudio.save(file_name, waveform, sample_rate)

        # Check if the file is empty and delete if so
        if torch.all(waveform == 0):
            os.remove(file_name)

if __name__ == '__main__':
    # Command line arguments: input_directory
    if len(sys.argv) != 2:
        print("Usage: python process_audio.py <input_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]

    if not os.path.isdir(input_directory):
        print("Error: Invalid input directory")
        sys.exit(1)

    # Process all audio files in the input directory
    process_directory(input_directory)
