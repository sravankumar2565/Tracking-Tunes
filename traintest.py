import librosa
import pandas as pd
import os
import numpy as np

def extract_features(file_name):
    try:
        # Load the audio file
        y, sr = librosa.load(file_name, sr=None)
        
        # Extract MFCC features and compute mean over time
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        # Extract Chroma features and compute mean over time
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        return mfccs_mean, chroma_mean  # Return them separately
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None, None

# Initialize an empty list to hold all the feature rows
data = []
# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Directories with audio files
for a in range(0, 39):
    if a < 10:
        audio_dir = f"00{a}"
    elif a < 100:
        audio_dir = f"0{a}"
    else:
        audio_dir = str(a)
    
    print(f"Processing directory: {audio_dir}/39")
    
    # Iterate through files in the directory
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        
        # Check if the file is an audio file and is not too small
        if file.endswith((".mp3", ".wav")) and os.path.getsize(file_path) > 2000:
            mfcc_features, chroma_features = extract_features(file_path)
            if mfcc_features is not None and chroma_features is not None:
                # Store MFCC and Chroma features in separate columns
                feature_dict = {
                    'file_name': file,
                    'mfcc': list(mfcc_features),  # Store MFCC features as a list
                    'chroma': list(chroma_features)  # Store Chroma features as a list
                }
                data.append(feature_dict)

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('song_features_mfcc_chroma_1.csv', index=False)

print("Feature extraction complete. Data saved to 'song_features_mfcc_chroma.csv'.")
