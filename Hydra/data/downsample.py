import os
import sys
import torchaudio

def resample_directory(input_directory, output_directory, new_sample_rate=16000):
    """
    Resample all files in the provided directory to a specified sample rate using torchaudio.
    """
    # Get list of all files in directory
    files = os.listdir(input_directory)

    for file_name in files:
        # Check if the file is .wav
        if not file_name.endswith('.wav'):
            continue

        # Construct full file path
        input_file_path = os.path.join(input_directory, file_name)
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(input_file_path)

        # Check if the audio needs to be resampled
        if sample_rate != new_sample_rate:
            # Resample the audio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
            resampled_waveform = resampler(waveform)

            # Save the resampled audio to the output directory
            output_file_path = os.path.join(output_directory, file_name)
            torchaudio.save(output_file_path, resampled_waveform, new_sample_rate, encoding="PCM_S", bits_per_sample=16)

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
