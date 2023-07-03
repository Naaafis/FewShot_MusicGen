import os
import sys
import torchaudio
from tqdm import tqdm

def resample_directory(input_directory, output_directory, new_sample_rate=16000):
    """
    Resample all files in the provided directory and its subdirectories to a specified sample rate using torchaudio.
    The audio is truncated to 1 second clips and each clip is saved as a separate file.
    """

    # Walk through input directory and its subdirectories
    all_files = [os.path.join(root, name) for root, dirs, files in os.walk(input_directory) for name in files]
    wav_files = [file for file in all_files if file.endswith(".wav")]
    for file_name in tqdm(wav_files, desc="Processing audio files", unit="file"):
        # Construct full file path
        input_file_path = file_name

        # Load the audio file
        waveform, sample_rate = torchaudio.load(input_file_path)

        # Check if the audio needs to be resampled
        if sample_rate != new_sample_rate:
            # Resample the audio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
            resampled_waveform = resampler(waveform)

            # Truncate to 1 second clips
            total_samples = resampled_waveform.shape[1]
            one_sec_samples = new_sample_rate

            for i in range(total_samples // one_sec_samples):
                start = i * one_sec_samples
                end = start + one_sec_samples

                # Construct the output file name
                output_file_name = f"{os.path.splitext(os.path.basename(file_name))[0]}_clip_{i}.wav"
                output_file_path = os.path.join(output_directory, output_file_name)

                # Save the 1-second clip to the output directory
                torchaudio.save(output_file_path, resampled_waveform[:, start:end], new_sample_rate, encoding="PCM_S", bits_per_sample=16)

if __name__ == '__main__':
    # Command line arguments: input_directory and output_directory
    if len(sys.argv) != 3:
        print("Usage: python resample_audio.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    if not os.path.isdir(input_directory):
        print("Error: Invalid input directory")
        sys.exit(1)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Resample all audio files in the input directory and save to the output directory
    resample_directory(input_directory, output_directory)
