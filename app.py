from flask import Flask, request, jsonify, render_template_string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import cv2

app = Flask(__name__)

# Load model and class labels
MODEL_PATH = 'hand_gesture_model.h5'
CLASS_LABELS_PATH = 'class_labels.npy'

# Gesture classes
GESTURES = ['call_me', 'fingers_crossed', 'okay', 'paper', 'peace', 
            'rock', 'rock_on', 'scissor', 'thumbs', 'up']

model = None
class_labels = {}

def load_model_and_labels():
    """Load the trained model and class labels"""
    global model, class_labels
    
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
    else:
        print("Warning: Model file not found. Please train the model first.")
    
    if os.path.exists(CLASS_LABELS_PATH):
        class_labels = np.load(CLASS_LABELS_PATH, allow_pickle=True).item()
        print("Class labels loaded successfully")
    else:
        # Use default labels if file doesn't exist
        class_labels = {i: gesture for i, gesture in enumerate(GESTURES)}
        print("Using default class labels")

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hand Gesture Recognition</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-section {
            text-align: center;
            margin: 30px 0;
        }
        input[type="file"] {
            display: none;
        }
        .upload-btn {
            background-color: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .upload-btn:hover {
            background-color: #45a049;
        }
        #preview {
            max-width: 300px;
            max-height: 300px;
            margin: 20px auto;
            display: none;
            border-radius: 5px;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            background-color: #e8f5e9;
            border-radius: 5px;
            display: none;
        }
        .gesture-name {
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }
        .confidence {
            font-size: 16px;
            color: #666;
            margin-top: 10px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖐️ Hand Gesture Recognition</h1>
        <div class="upload-section">
            <label for="file-upload" class="upload-btn">
                Choose Image
            </label>
            <input id="file-upload" type="file" accept="image/*" onchange="previewImage(event)">
            <br><br>
            <img id="preview" alt="Preview">
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing gesture...</p>
        </div>
        <div class="result" id="result">
            <div class="gesture-name" id="gesture-name"></div>
            <div class="confidence" id="confidence"></div>
        </div>
    </div>

    <script>
        function previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    
                    // Send image to server for prediction
                    predictGesture(file);
                }
                reader.readAsDataURL(file);
            }
        }

        async function predictGesture(file) {
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            loading.style.display = 'block';
            result.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                loading.style.display = 'none';
                result.style.display = 'block';
                
                document.getElementById('gesture-name').textContent = 
                    `Gesture: ${data.gesture.replace('_', ' ').toUpperCase()}`;
                document.getElementById('confidence').textContent = 
                    `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            } catch (error) {
                loading.style.display = 'none';
                alert('Error predicting gesture. Please try again.');
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Render the home page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict hand gesture from uploaded image"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        img_array = image.img_to_array(image.load_img(file, target_size=(128, 128)))
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get gesture name
        gesture_name = class_labels.get(predicted_class, GESTURES[predicted_class])
        
        return jsonify({
            'gesture': gesture_name,
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    load_model_and_labels()
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=True)
