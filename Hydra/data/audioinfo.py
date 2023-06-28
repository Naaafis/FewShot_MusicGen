import os
import sys
import torchaudio

def get_audio_info(directory):
    """
    Get and print the sample rate, bit depth, and length of all audio files in a directory.
    """
    # Get list of all files in directory
    files = os.listdir(directory)

    for file_name in files:
        # Check if the file is .wav
        if not file_name.endswith('.wav'):
            continue

        # Construct full file path
        file_path = os.path.join(directory, file_name)

        # Get audio file info
        info = torchaudio.info(file_path)

        # Calculate audio length in seconds
        length_in_seconds = info.num_frames / info.sample_rate

        # Print information
        print(f'File: {file_name}')
        print(f'Sample rate: {info.sample_rate} Hz')
        print(f'Bit depth: {info.bits_per_sample} bits')
        print(f'Segment length: {length_in_seconds} seconds')
        print('----------')

if __name__ == '__main__':
    # Command line argument: directory
    if len(sys.argv) != 2:
        print("Usage: python audio_info.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print("Error: Invalid directory")
        sys.exit(1)

    # Get and print audio info for all files in the directory
    get_audio_info(directory)
