# Hand Gesture Recognition

A deep learning model for recognizing hand gestures using Convolutional Neural Networks (CNN).

## Features

- **10 Gesture Classes**: call_me, fingers_crossed, okay, paper, peace, rock, rock_on, scissor, thumbs, up
- **CNN Architecture**: 4 convolutional layers with dropout for better generalization
- **Web Interface**: Easy-to-use Flask web application for real-time predictions
- **Data Augmentation**: Rotation, shifting, and horizontal flipping for robust training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gyanankur23/PRODIGY_ML_04.git
cd PRODIGY_ML_04
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare dataset:
- Place your hand gesture images in `HandGesture/images/` directory
- Create subdirectories for each gesture class
- Example structure:
```
HandGesture/images/
├── call_me/
├── fingers_crossed/
├── okay/
├── paper/
├── peace/
├── rock/
├── rock_on/
├── scissor/
├── thumbs/
└── up/
```

## Usage

### Training the Model

Train the model on your dataset:
```bash
python train_model.py
```

This will:
- Train a CNN model on your gesture images
- Save the trained model as `hand_gesture_model.h5`
- Save class labels as `class_labels.npy`

### Running the Web Application

Start the Flask web server:
```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`

Upload an image of a hand gesture to see the prediction.

## Model Architecture

```
Conv2D (32 filters) -> MaxPooling2D
Conv2D (64 filters) -> MaxPooling2D
Conv2D (128 filters) -> MaxPooling2D
Conv2D (128 filters) -> MaxPooling2D
Flatten -> Dense (512) -> Dropout (0.5)
Dense (10) - Softmax
```

## API Endpoints

- `GET /` - Web interface for gesture prediction
- `POST /predict` - Upload image and get prediction
- `GET /health` - Health check endpoint

## Deployment

### Quick Deployment (Local)

1. Train the model: `python train_model.py`
2. Run the app: `python app.py`
3. Open http://localhost:5000

### Cloud Deployment Options

- **Heroku**: Deploy using the Procfile and requirements.txt
- **Render**: Connect GitHub repository for automatic deployment
- **Railway**: Simple deployment from GitHub
- **PythonAnywhere**: Easy web hosting for Python apps

## Requirements

- Python 3.8+
- TensorFlow 2.15.0
- NumPy 1.24.3
- OpenCV 4.8.1
- Pillow 10.1.0
- Flask 3.0.0
- scikit-learn 1.3.2

## License

This project is part of the PRODIGY ML internship program.
