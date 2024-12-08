# Audio Genre Classification Project

## Project Overview
This project implements an audio genre classification system using Convolutional Neural Networks (CNN) and spectrograms. Inspired by the Shazam algorithm, the project aims to classify audio files into different music genres.

## Setup and Installation

### Prerequisites
- Python 3.12+
- pip (Python package manager)

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/projet_classification_audio.git
cd projet_classification_audio
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Preparation
- Download the GTZAN Genre Dataset
- Place the dataset in a directory (e.g., `data/genres`)
- Ensure the directory structure is:
```
data/genres/
│
├── blues/
│   ├── blues.00000.wav
│   ├── blues.00001.wav
│   └── ...
│
├── classical/
│   ├── classical.00000.wav
│   ├── classical.00001.wav
│   └── ...
└── ...
```

### Training the Model
Run the training script:
```bash
python src/model/train.py
```
- The trained model will be saved in `models/genre_classifier.h5`

### Running the Streamlit App
```bash
streamlit run src/app/streamlit_app.py
```

## Project Structure
```
projet_classification_audio/
│
├── src/
│   ├── model/
│   │   ├── model.py         # CNN model definition
│   │   └── train.py         # Training script
│   │
│   ├── api/
│   │   └── prediction.py    # Prediction logic
│   │
│   └── app/
│       └── streamlit_app.py # Streamlit interface
│
├── models/
│   └── genre_classifier.h5  # Saved trained model
│
└── requirements.txt
```

## Features
- Audio file genre classification
- Mel-spectrogram visualization
- Confidence score for predictions
- Streamlit-based interactive interface

## Notes
- Supported Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- Model performance may vary based on the training dataset

## Extensions (Optional)
- Improve model by collecting more diverse audio samples
- Implement data augmentation techniques
- Experiment with different neural network architectures