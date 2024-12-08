from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import os

# Initialize Flask app
app = Flask(__name__)

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
    
    def extract_melspectrogram(self, audio_data, max_pad_len=174):
        """
        Extract Mel-spectrogram features from an audio file
        
        Args:
            audio_data (numpy.ndarray): Audio data array
            max_pad_len (int): Maximum length to pad or truncate spectrograms
        
        Returns:
            numpy.ndarray: Preprocessed mel-spectrogram
        """
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=22050, 
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
    
    def predict_genre(self, audio_data):
        """
        Predict the genre of an audio file
        
        Args:
            audio_data (numpy.ndarray): Audio data array
        
        Returns:
            dict: Prediction results with genre and confidence
        """
        # Preprocess the audio
        preprocessed_audio = self.extract_melspectrogram(audio_data)
        
        # Make prediction
        predictions = self.model.predict(preprocessed_audio)
        
        # Get the predicted genre
        predicted_genre_idx = np.argmax(predictions[0])
        predicted_genre = self.genres[predicted_genre_idx]
        confidence = float(predictions[0][predicted_genre_idx])

        print(f"Predicted genre: {predicted_genre} (Confidence: {confidence}) - All predictions: {predictions[0]}")
        
        return {
            'genre': predicted_genre,
            'confidence': confidence,
            'all_predictions': {
                genre: float(conf) for genre, conf in zip(self.genres, predictions[0])
            }
        }

# Initialize the classifier
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
model_path = '/app/models/best_model.keras'  # Change this to your actual model path
classifier = AudioClassifier(model_path=model_path, genres=genres)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for genre prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read the file as an audio file
        audio_data, sr = librosa.load(file, sr=22050)
        
        # Get prediction results
        result = classifier.predict_genre(audio_data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)