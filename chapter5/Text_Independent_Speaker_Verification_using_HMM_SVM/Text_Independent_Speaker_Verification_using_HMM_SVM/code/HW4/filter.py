import os
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
import noisereduce as nr
import csv 

def get_filenames_by_lowest_subfolder(root_folder, x):
    file_dict = {}
    
    # Get the subdirectories of the root folder and sort them numerically
    subfolders = [subfolder for subfolder in os.listdir(root_folder) 
                  if os.path.isdir(os.path.join(root_folder, subfolder))]
    
    # Sort subfolders numerically if they are numbers, otherwise as strings
    subfolders.sort(key=lambda name: int(name) if name.isdigit() else name)
    
    # Limit to the first x subfolders
    subfolders = subfolders[:x]
    
    # Walk through each selected subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)
        
        for dirpath, _, filenames in os.walk(subfolder_path):
            if filenames:  # If there are files in the directory
                # Get the last folder in the current path as the lowest subfolder
                lowest_subfolder = os.path.basename(dirpath)
                
                # Initialize the list for the lowest subfolder if it doesn't exist
                if lowest_subfolder not in file_dict:
                    file_dict[lowest_subfolder] = []
                
                # Add files to the respective lowest subfolder
                for filename in filenames:
                    if filename.endswith(".flac"):
                        full_file_path = os.path.join(dirpath, filename)
                        file_dict[lowest_subfolder].append(full_file_path)
    
    return file_dict

def estimate_snr_weighted(audio_signal, base_sigma=1.2, min_segment_length=500, max_segment_length=1500, adaptive_factor=0.01):
    """
    Estimating the Signal-to-Noise Ratio (SNR) of a given speech signal using weighted adaptive smoothing,
    dynamic segment lengths, and signal power scaling for higher SNR.
    """
    if len(audio_signal.shape) > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    normalized_signal = audio_signal / np.max(np.abs(audio_signal))
    power_signal = np.mean(normalized_signal ** 2)
    max_signal_value = np.max(np.abs(normalized_signal))
    adaptive_noise_threshold = adaptive_factor * max_signal_value

    smoothed_signal = np.zeros_like(normalized_signal)
    total_samples = len(normalized_signal)
    pos = 0

    while pos < total_samples:
        segment_length = np.random.randint(min_segment_length, max_segment_length)
        end = min(pos + segment_length, total_samples)
        segment = normalized_signal[pos:end]
        segment_power = np.abs(segment).mean()

        if segment_power < adaptive_noise_threshold:
            local_sigma = base_sigma + 3
        else:
            local_sigma = base_sigma

        smoothed_signal[pos:end] = gaussian_filter1d(segment, sigma=local_sigma)
        pos = end

    noise = normalized_signal - smoothed_signal
    power_noise = np.mean(noise ** 2)
    snr = 10 * np.log10(power_signal / power_noise)

    return snr

def filter_and_snr(audio_signal, sample_rate):
    
    snr_estimated = estimate_snr_weighted(audio_signal)
    denoised_audio_signal = nr.reduce_noise(y=audio_signal, sr=sample_rate, prop_decrease=0.8)
    snr_estimated_cleaned = estimate_snr_weighted(denoised_audio_signal)

    return(snr_estimated, snr_estimated_cleaned, denoised_audio_signal)

def main():
    root_directory = r"C:\Computer Science Programs\Fall_2024\EE502_BioMed\project\data\LibriSpeech\train-clean-100"
    save_to_dir = r"C:\Computer Science Programs\Fall_2024\EE502_BioMed\project\data\cleaned"
    SAMPLE_RATE = 16000

    x = 251  # Change this to the number of subfolders you want to limit to
    files_by_lowest_subfolder = get_filenames_by_lowest_subfolder(root_directory, x)

    print(len(files_by_lowest_subfolder))

    for lowest_subfolder, files in files_by_lowest_subfolder.items():
        print(f"Lowest Subfolder: {lowest_subfolder}")
        
        
        denoised_file_paths = []
        densoised_to_save = []
        
        snr_data = []

        for file in files:
            relative_path = os.path.relpath(file, root_directory)
            filename, _ = os.path.splitext(relative_path)
            save_to = os.path.join(save_to_dir, filename + ".wav")
            audio_signal, _ = sf.read(file)
            
            snr_estimated, snr_estimated_cleaned, denoised_audio_signal = filter_and_snr(audio_signal, SAMPLE_RATE)
            # print(f"SNR original: {snr_estimated} | SNR cleaned: {snr_estimated_cleaned}")
            snr_data.append([filename, snr_estimated, snr_estimated_cleaned])

            denoised_file_paths.append(save_to)
            densoised_to_save.append(denoised_audio_signal)
            
        print(f"writing snr for: {lowest_subfolder}")
        
        with open('../../data/output/snr_data_all.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(snr_data)

        # for save_to, denoised_audio_signal in zip(denoised_file_paths, densoised_to_save):
        #     sf.write(save_to, np.asarray(denoised_audio_signal), SAMPLE_RATE)

if __name__=="__main__":
    main()