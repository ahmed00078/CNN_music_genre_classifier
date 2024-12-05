import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import tempfile
import os

# Import the prediction module
from src.api.prediction import AudioClassifier

def plot_mel_spectrogram(audio_path):
    """
    Plot mel-spectrogram of the input audio
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        matplotlib.figure.Figure: Mel-spectrogram visualization
    """
    # Load audio file
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sample_rate, 
        n_mels=128,
        fmax=8000
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spec_db, 
        sr=sample_rate, 
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    
    return plt.gcf()

def main():
    """
    Streamlit app main function
    """
    st.title('🎵 Audio Genre Classifier')
    
    # Initialize genres and model
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    # Load the model
    try:
        classifier = AudioClassifier(
            model_path='models/genre_classifier.h5', 
            genres=genres
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file", 
        type=['wav', 'mp3', 'ogg'],
        help="Upload a short audio clip to classify its genre"
    )
    
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            # Display audio player
            st.audio(uploaded_file, format='audio/wav')
            
            # Predict genre
            prediction = classifier.predict_genre(temp_file_path)
            
            # Display prediction results
            st.subheader("Prediction Results")
            st.write(f"**Predicted Genre:** {prediction['genre']}")
            st.write(f"**Confidence:** {prediction['confidence']*100:.2f}%")
            
            # Confidence bar chart
            st.subheader("Genre Confidence Scores")
            confidence_data = prediction['all_predictions']
            
            # Create a bar chart of confidence scores
            fig, ax = plt.subplots(figsize=(10, 6))
            genres = list(confidence_data.keys())
            confidences = list(confidence_data.values())
            
            ax.bar(genres, [conf*100 for conf in confidences])
            ax.set_ylabel('Confidence (%)')
            ax.set_title('Genre Confidence Scores')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Visualize Mel-spectrogram
            st.subheader("Audio Mel-spectrogram")
            mel_fig = plot_mel_spectrogram(temp_file_path)
            st.pyplot(mel_fig)
        
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    # Sidebar information
    st.sidebar.title("About the App")
    st.sidebar.info(
        "This app uses a Convolutional Neural Network (CNN) "
        "to classify audio files into music genres. "
        "Upload a short audio clip to get started!"
    )
    
    st.sidebar.subheader("Supported Genres")
    st.sidebar.write(", ".join(genres))

if __name__ == "__main__":
    main()