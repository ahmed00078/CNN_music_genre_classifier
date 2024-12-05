import os
import numpy as np
import librosa

def load_audio_files(data_path='./data/genres_original'):
    """Load audio files and their labels."""
    X, y = [], []
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        print(f"Processing genre: {genre}, path: {genre_path}")
        
        if not os.path.exists(genre_path):
            print(f"Warning: Directory {genre_path} does not exist.")
            continue
        
        for file in os.listdir(genre_path):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_path, file)
                try:
                    audio, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
                    
                    # Padding or truncating to ensure fixed length
                    required_length = 30 * sr  # 30 seconds
                    if len(audio) < required_length:
                        audio = np.pad(audio, (0, required_length - len(audio)), mode='constant')
                    elif len(audio) > required_length:
                        audio = audio[:required_length]
                    
                    X.append(audio)
                    y.append(genres.index(genre))
                except Exception as e:
                    print(f"Skipping {file_path} due to error: {e}")
    
    print(f"Total files loaded: {len(X)}")
    return np.array(X), np.array(y)

def extract_melspectrogram(audio, sr=22050, n_mels=128, hop_length=512):
    """Extract Mel spectrogram features with fixed width."""
    mel_spec = librosa.feature.melspectrogram(y=audio, 
                                              sr=sr, 
                                              n_mels=n_mels,
                                              hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Fix the width of the spectrogram
    fixed_width = int(30 * sr / hop_length)  # Calculate expected width for 30 seconds
    if mel_spec_db.shape[1] < fixed_width:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, fixed_width - mel_spec_db.shape[1])), mode='constant')
    elif mel_spec_db.shape[1] > fixed_width:
        mel_spec_db = mel_spec_db[:, :fixed_width]
    
    return mel_spec_db

def prepare_dataset(X, y, n_mels=128):
    """Prepare dataset for CNN."""
    X_processed = []
    for i, audio in enumerate(X):
        try:
            mel_spec = extract_melspectrogram(audio, n_mels=n_mels)
            X_processed.append(mel_spec)
        except Exception as e:
            print(f"Error processing audio {i}: {e}")
    
    if not X_processed:
        raise ValueError("No valid data to process. Check your input audio files.")

    X_processed = np.array(X_processed)
    print(f"Processed dataset shape: {X_processed.shape}")
    
    # Add channel dimension for CNN
    X_processed = X_processed[X_processed.shape[0], 
                              X_processed.shape[1], 
                              X_processed.shape[2], 
                              np.newaxis]  # Shape becomes (samples, height, width, channels)
    return X_processed, np.array(y)
