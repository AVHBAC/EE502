import librosa
import os
import pickle 

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

                # Initialize the dictionary for the subfolder if it doesn't exist
                if subfolder not in file_dict:
                    file_dict[subfolder] = {}

                # Initialize the list for the lowest subfolder if it doesn't exist
                if lowest_subfolder not in file_dict[subfolder]:
                    file_dict[subfolder][lowest_subfolder] = []
                
                # Add files to the respective lowest subfolder
                for filename in filenames:
                    if filename.endswith(".wav"):
                        full_file_path = os.path.join(dirpath, filename)
                        file_dict[subfolder][lowest_subfolder].append(full_file_path)
        
    return file_dict

def extract_mfcc(file_dictionary):
    mfcc_dict = {}

    for reader in file_dictionary.keys():
        
        if reader not in mfcc_dict:
            mfcc_dict[reader] = {}

        for chapter in file_dictionary[reader].keys():
            
            if chapter not in mfcc_dict[reader]:
                mfcc_dict[reader][chapter] = []

            for audio_path in file_dictionary[reader][chapter]:
                signal, sample_rate = librosa.load(audio_path, sr=None)
                mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=5)
                mfcc_dict[reader][chapter].append(mfccs)

    with open("../../../data/extracted_features/mfccs/mfccs_dict_5.pickle", "wb") as file:
        pickle.dump(mfcc_dict, file)

def main():
    root = r"C:\Computer Science Programs\Fall_2024\EE502_BioMed\project\data\cleaned"
    file_dictionary = get_filenames_by_lowest_subfolder(root)
    extract_mfcc(file_dictionary)

if __name__=="__main__":
    main()