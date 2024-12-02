# Bird Species Identification Using Deep Learning

## Overview
This project implements a Convolutional Neural Network (CNN) for automatic bird species classification using audio recordings. By converting bird sounds into spectrograms and applying deep learning techniques, we've developed a robust model for ecological monitoring and wildlife conservation.

## Features

### Data Collection
- Dataset: 2161 audio files
- 114 distinct bird species
- Sourced from Kaggle: [Bird Sound Dataset](https://www.kaggle.com/datasets/soumendraprasad/sound-of-114-species-of-birds-till-2022)

### Deployment
- Streamlit web interface
- Support for MP3 and WAV uploads
- Real-time bird species classification

## Project Objectives
- Develop a CNN-based classification model using TensorFlow
- Improve bird species identification accuracy in noisy conditions
- Create an automated solution for ecological monitoring
- Support conservation activities through technology

## Key Technologies
- Python
- TensorFlow
- Convolutional Neural Network (CNN)
- Keras
- Streamlit
- Librosa
- OpenCV
- NumPy

## Methodology
### Data Preprocessing
- Resampling audio files to standard frequency (16 kHz)
- Converting audio to spectrograms using Short-Time Fourier Transform (STFT)
- Applying noise reduction and normalization techniques
- Implementing data augmentation (time stretching, pitch shifting)

### Model Architecture
- 3 Convolutional layers (32, 64, 128 filters)
- Max-pooling layers for dimensionality reduction
- 2 Fully connected layers
- Softmax output layer
- Adam optimizer (learning rate: 0.001)
- Dropout regularization

## Performance Metrics
- Accuracy: 0.7875
- Precision: 0.7919
- Recall: 0.6875
- F1-Score: 0.7129

## Challenges Addressed
- Handling similar bird call patterns
- Managing background noise in recordings
- Accounting for variations in bird vocalizations

## Potential Applications
- Biodiversity monitoring
- Population tracking
- Habitat health assessment
- Conservation research

## Installation

### Prerequisites
```bash
pip install tensorflow
pip install scikit-learn
pip install opencv-python
pip install librosa
pip install numpy
pip install pandas
pip install streamlit
pip install streamlit_extras
```

## Usage

1. Clone the repository
```bash
git clone https://github.com/your-repo/Bird-Sound-Classification.git
```

2. Install requirements
```bash
pip install -r requirements.txt
```

3. Run Streamlit Application
```bash
streamlit run main.py
```

4. Access the app at `http://localhost:8501`


## License
This project is licensed under the MIT License. Please review the LICENSE file for more details.

## Contact
- Email: villwin11@gmail.com
- LinkedIn: [**Dhanush Saravanan**](https://www.linkedin.com/in/dhanush-saravanan-148857268?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
 

## Future Work
- Expand dataset with more bird species
- Improve model generalization
- Reduce computational complexity
- Real-time bird sound classification
