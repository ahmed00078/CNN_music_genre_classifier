import numpy as np
import tensorflow as tf
import librosa

class AudioClassifier:
    def __init__(self, model_path, genres):
        """
        Initialize the audio classifier
        
        Args:
            model_path (str): Path to the saved model
            genres (list): List of genre labels
        """
        self.model = tf.keras.models.load_model(model_path)
        self.genres = genres
    
    def extract_melspectrogram(self, file_path, max_pad_len=174):
        """
        Extract Mel-spectrogram features from an audio file
        
        Args:
            file_path (str): Path to the audio file
            max_pad_len (int): Maximum length to pad or truncate spectrograms
        
        Returns:
            numpy.ndarray: Preprocessed mel-spectrogram
        """
        # Load audio file
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
        
        # Reshape for model input
        mel_spec_db = mel_spec_db.reshape(1, mel_spec_db.shape[0], mel_spec_db.shape[1], 1)
        
        return mel_spec_db
    
    def predict_genre(self, audio_path):
        """
        Predict the genre of an audio file
        
        Args:
            audio_path (str): Path to the audio file
        
        Returns:
            dict: Prediction results with genre and confidence
        """
        # Preprocess the audio
        preprocessed_audio = self.extract_melspectrogram(audio_path)
        
        # Make prediction
        predictions = self.model.predict(preprocessed_audio)
        
        # Get the predicted genre
        predicted_genre_idx = np.argmax(predictions[0])
        predicted_genre = self.genres[predicted_genre_idx]
        confidence = float(predictions[0][predicted_genre_idx])
        
        return {
            'genre': predicted_genre,
            'confidence': confidence,
            'all_predictions': {
                genre: float(conf) for genre, conf in zip(self.genres, predictions[0])
            }
        }

# Example usage
if __name__ == "__main__":
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    classifier = AudioClassifier(
        model_path='../models/genre_classifier.h5', 
        genres=genres
    )
    
    # Test prediction
    test_audio_path = 'path/to/test/audio/file.wav'
    result = classifier.predict_genre(test_audio_path)
    print(result)