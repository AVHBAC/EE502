import os
import librosa
import random
import multiprocessing

import noisereduce as nr
import soundfile as sf
import numpy as np

from pathlib import Path

def get_speakers(meta_data):
    male_speakers = []
    female_speakers = []
    random.seed(42)

    with open(meta_data, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Skip lines starting with a semicolon (comments)
            if line.strip().startswith(";") or not line.strip():
                continue
            
            # Split the line by pipe (|) and strip extra spaces
            fields = [field.strip() for field in line.split('|')]
            
            # Ensure the line has all required fields
            if len(fields) >= 5:
                reader_id = fields[0]
                gender = fields[1]
                subset = fields[2]
                length = float(fields[3])

                # Add to appropriate list based on conditions
                if subset == "train-clean-100":
                    if gender == "M":
                        male_speakers.append(reader_id)
                    elif gender == "F":
                        female_speakers.append(reader_id)


    # male_speakers_sample = random.sample(male_speakers, min(50, len(male_speakers)))
    # female_speakers_sample = random.sample(female_speakers, min(50, len(female_speakers)))

    # samples = male_speakers_sample
    # samples.extend(female_speakers_sample)

    samples = male_speakers 
    samples.extend(female_speakers)

    # search for already cleaned files 
    already_cleaned_folder = "../../../data/cleaned_combined_v2"

    already_cleaned_names = [
        os.path.splitext(file)[0]  # Split file into name and extension, take the name
        for file in os.listdir(already_cleaned_folder)  # List files in the folder
        if os.path.isfile(os.path.join(already_cleaned_folder, file))  # Ensure it's a file
    ]

    samples = [names for names in samples if names not in already_cleaned_names]

    return samples

def clean_and_combine(speakers, data_root="../../../data/LibriSpeech/train-clean-100/") -> None:
    for speaker in speakers:
        denoised_files = []
        if not os.path.exists(f"../../../data/cleaned_combined_v2/{speaker}.wav"):
            for root, _, files in os.walk(f"{data_root}{speaker}"):
                for file in files:
                    if file.endswith('.flac'):
                        input_file_path = Path(root) / file
                        audio_signal, sample_rate = librosa.load(input_file_path, sr=None)
                        denoised_audio_signal = nr.reduce_noise(y=audio_signal, sr=sample_rate, prop_decrease=0.8)
                        denoised_files.append(denoised_audio_signal)

            combined_audio = np.concatenate(denoised_files)
            sf.write(f"../../../data/cleaned_combined_v2/{speaker}.wav", combined_audio, sample_rate)
            print(f"speaker {speaker} completed")
        else:
            print(f"file for {speaker} already exists skipping")

def chunk_list(to_chunk, num_chunks=4):
    avg_chunk_size = len(to_chunk) // num_chunks
    remainder = len(to_chunk) % num_chunks

    chunks = []
    start_index = 0

    for i in range(num_chunks):
        # Add 1 to the chunk size until the remainder is used up
        end_index = start_index + avg_chunk_size + (1 if i < remainder else 0)
        chunks.append(to_chunk[start_index:end_index])
        start_index = end_index
    
    return chunks

def main():
    cores = 6
    file_path = "../../../data/LibriSpeech/SPEAKERS.TXT"

    samples = get_speakers(file_path)

    chunks = chunk_list(samples, num_chunks=cores)

    with multiprocessing.Pool(processes=cores) as pool:
        pool.map(clean_and_combine, chunks)

if __name__ == "__main__":
    main()