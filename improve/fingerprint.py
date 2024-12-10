import numpy as np
import librosa
import hashlib
import pandas as pd
import streamlit as st
from typing import List, Tuple, Dict
import os
import pickle

class AudioFingerprinter:
    def __init__(self, target_sample_rate=44100, segment_duration=5):
        """
        Initialize the audio fingerprinting system
        
        Args:
        - target_sample_rate: Standard sampling rate for audio
        - segment_duration: Duration of audio segments to analyze (in seconds)
        """
        self.target_sample_rate = target_sample_rate
        self.segment_duration = segment_duration
        self.song_database = {}
        
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Preprocess audio file
        
        Args:
        - audio_path: Path to audio file
        
        Returns:
        - Preprocessed mono audio signal
        """

        print(f"Processing {audio_path}...")

        # Load audio file
        audio, sample_rate = librosa.load(
            audio_path, 
            sr=self.target_sample_rate, 
            mono=True
        )
        return audio
    
    def generate_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Generate spectrogram from audio signal
        
        Args:
        - audio: Preprocessed mono audio signal
        
        Returns:
        - Spectrogram representing audio frequencies
        """

        print(f"Generating spectrogram...")

        # Compute spectrogram
        spectrogram = librosa.stft(audio)
        spectrogram_db = librosa.amplitude_to_db(
            np.abs(spectrogram), 
            ref=np.max
        )
        return spectrogram_db
    
    def find_peaks(self, spectrogram: np.ndarray, neighborhood_size=5) -> List[Tuple[int, int]]:
        """
        Find peak points in the spectrogram
        
        Args:
        - spectrogram: Spectrogram of the audio
        - neighborhood_size: Size of local neighborhood to compare
        
        Returns:
        - List of peak points (time, frequency)
        """

        print(f"Finding peaks...")
        
        # Find local peaks in the spectrogram
        peaks = []
        for t in range(neighborhood_size, spectrogram.shape[1] - neighborhood_size):
            for f in range(neighborhood_size, spectrogram.shape[0] - neighborhood_size):
                is_peak = True
                central_value = spectrogram[f, t]
                
                # Check local neighborhood
                for dt in range(-neighborhood_size, neighborhood_size + 1):
                    for df in range(-neighborhood_size, neighborhood_size + 1):
                        if dt == 0 and df == 0:
                            continue
                        if spectrogram[f + df, t + dt] > central_value:
                            is_peak = False
                            break
                    if not is_peak:
                        break
                
                if is_peak:
                    peaks.append((t, f))
        
        return peaks
    
    def generate_hash(self, peak1: Tuple[int, int], peak2: Tuple[int, int]) -> str:
        """
        Generate a hash from two peak points
        
        Args:
        - peak1: First peak point (time, frequency)
        - peak2: Second peak point (time, frequency)
        
        Returns:
        - Unique hash representing the relationship between peaks
        """
        # Combine peak information to create a unique hash
        hash_input = f"{peak1[0]}|{peak1[1]}|{peak2[0]}|{peak2[1]}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def create_fingerprint(self, audio_path: str, song_name: str):
        """
        Create a fingerprint for a song
        
        Args:
        - audio_path: Path to audio file
        - song_name: Name of the song
        """

        print(f"Processing {song_name}...")

        # Preprocess audio
        audio = self.preprocess_audio(audio_path)
        
        # Generate spectrogram
        spectrogram = self.generate_spectrogram(audio)
        
        # Find peaks
        peaks = self.find_peaks(spectrogram)
        
        # Generate hashes
        song_hashes = {}
        for i in range(len(peaks)):
            for j in range(1, 5):  # Look at next few peaks
                if i + j < len(peaks):
                    hash_key = self.generate_hash(peaks[i], peaks[i+j])
                    song_hashes[hash_key] = peaks[i][0]  # Store time of first peak
        
        # Store in song database
        self.song_database[song_name] = song_hashes
    
    def match_audio(self, query_path: str, top_n=5) -> List[Tuple[str, float]]:
        """
        Match a query audio to songs in the database
        
        Args:
        - query_path: Path to query audio file
        - top_n: Number of top matches to return
        
        Returns:
        - List of matched songs with confidence scores
        """
        # Preprocess query audio
        query_audio = self.preprocess_audio(query_path)
        query_spectrogram = self.generate_spectrogram(query_audio)
        query_peaks = self.find_peaks(query_spectrogram)
        
        # Generate hashes for query
        query_hashes = {}
        for i in range(len(query_peaks)):
            for j in range(1, 5):
                if i + j < len(query_peaks):
                    hash_key = self.generate_hash(query_peaks[i], query_peaks[i+j])
                    query_hashes[hash_key] = query_peaks[i][0]
        
        # Match hashes
        matches = {}
        for song_name, song_hashes in self.song_database.items():
            match_count = 0
            for hash_key, query_time in query_hashes.items():
                if hash_key in song_hashes:
                    match_count += 1
            
            # Calculate match confidence
            match_percentage = match_count / len(query_hashes) * 100
            matches[song_name] = match_percentage
        
        # Sort matches by confidence
        sorted_matches = sorted(
            matches.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_matches[:top_n]

def build_database(fingerprinter: AudioFingerprinter, music_directory: str):
    """
    Build a database of song fingerprints
    
    Args:
    - fingerprinter: AudioFingerprinter instance
    - music_directory: Directory containing music files
    """

    print(f"Processing music files in {music_directory}...")

    # Iterate through music files
    for filename in os.listdir(music_directory):
        if filename.endswith(('.mp3', '.wav', '.ogg')):
            file_path = os.path.join(music_directory, filename)
            song_name = os.path.splitext(filename)[0]
            
            try:
                fingerprinter.create_fingerprint(file_path, song_name)
                print(f"Processed: {song_name}")
            except Exception as e:
                print(f"Error processing {song_name}: {e}")

def save_database(fingerprinter: AudioFingerprinter, database_path: str):
    """
    Save the song database to a file
    
    Args:
    - fingerprinter: AudioFingerprinter instance
    - database_path: Path to save the database
    """
    with open(database_path, 'wb') as f:
        pickle.dump(fingerprinter.song_database, f)

def load_database(fingerprinter: AudioFingerprinter, database_path: str):
    """
    Load a previously saved song database
    
    Args:
    - fingerprinter: AudioFingerprinter instance
    - database_path: Path to load the database from
    """
    with open(database_path, 'rb') as f:
        fingerprinter.song_database = pickle.load(f)

# Streamlit Application
def streamlit_audio_recognizer():
    """
    Streamlit app for audio recognition
    """
    st.title("🎵 Music Recognition App")
    
    # Initialize fingerprinter
    fingerprinter = AudioFingerprinter()
    
    # Load existing database
    database_path = 'song_database.pkl'
    if os.path.exists(database_path):
        load_database(fingerprinter, database_path)
        st.success(f"Loaded database with {len(fingerprinter.song_database)} songs")
    else:
        st.warning("No existing database found. Build a database first!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an audio file to recognize", 
        type=['wav', 'mp3', 'ogg']
    )
    
    if uploaded_file is not None:
        # Temporarily save the file
        with open("temp_query.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Play the uploaded audio
        st.audio(uploaded_file, format='audio/wav')
        
        # Match the audio
        with st.spinner('Recognizing song...'):
            try:
                matches = fingerprinter.match_audio("temp_query.wav")
                
                # Display results
                st.header("Recognition Results")
                for song, confidence in matches:
                    st.write(f"- {song} (Confidence: {confidence:.2f}%)")
            
            except Exception as e:
                st.error(f"Error in recognition: {e}")

def main():
    """
    Main execution function
    """
    # Create fingerprinter
    fingerprinter = AudioFingerprinter()

    print("Building song database...")
    
    # Build database (do this once)
    music_directory = '../data/genres_original/blues'
    build_database(fingerprinter, music_directory)
    
    # Save database
    save_database(fingerprinter, 'song_database.pkl')
    
    # Optional: Run Streamlit app
    # streamlit_audio_recognizer()

if __name__ == "__main__":
    main()