import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

import librosa
import pandas as pd

from cnn_model import create_audio_cnn_model

def extract_melspectrogram(file_path, max_pad_len=174):
    """
    Extract Mel-spectrogram features from an audio file
    
    Args:
        file_path (str): Path to the audio file
        max_pad_len (int): Maximum length to pad or truncate spectrograms
    
    Returns:
        numpy.ndarray: Mel-spectrogram representation
    """
    # Load audio file
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sample_rate, 
            n_mels=128,
            fmax=8000
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or truncate
        if mel_spec_db.shape[1] > max_pad_len:
            mel_spec_db = mel_spec_db[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, 
                                 pad_width=((0, 0), (0, pad_width)), 
                                 mode='constant')
        
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def prepare_dataset(data_path, genres):
    """
    Prepare dataset by extracting features and labels
    
    Args:
        data_path (str): Path to the dataset
        genres (list): List of genres to classify
    
    Returns:
        tuple: X (features), y (labels)
    """
    X = []
    y = []
    
    for genre_idx, genre in enumerate(genres):
        genre_path = os.path.join(data_path, genre)
        print(f"Looking for files in: {genre_path}")
        
        for audio_file in os.listdir(genre_path):
            file_path = os.path.join(genre_path, audio_file)
            
            mel_spec = extract_melspectrogram(file_path)
            
            if mel_spec is not None:
                X.append(mel_spec)
                y.append(genre_idx)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for CNN input (add channel dimension)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # One-hot encode labels
    y = to_categorical(y)
    
    return X, y

def train_audio_classifier(data_path, genres, output_model_path):
    """
    Train audio genre classification model
    
    Args:
        data_path (str): Path to the dataset
        genres (list): List of genres
        output_model_path (str): Path to save the trained model
    """
    # Prepare dataset
    X, y = prepare_dataset(data_path, genres)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create the model
    model = create_audio_cnn_model(
        input_shape=(X_train.shape[1], X_train.shape[2], 1), 
        num_classes=len(genres)
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        output_model_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint, early_stop]
    )
    try:
        # Save the training history
        history_df = pd.DataFrame(history.history)
        history_path = output_model_path.replace('.h5', '_history.csv')
        history_df.to_csv(history_path, index=False)
    except Exception as e:
        print(f"Error saving training history: {e}")

    try:
        # Save the model
        model.save(output_model_path)
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    return history

# Example usage
if __name__ == "__main__":
    # Define genres (adjust based on your dataset)
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Paths
    # data_path = 'data/genres_original'
    output_model_path = '../../models/best_model.h5'
    
    data_path = "C:/Users/hp/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Ensit-Info/S5/ML/mini projet/music_genre_classifier/data/genres_original"
    
    # Train the model
    train_audio_classifier(data_path, genres, output_model_path)