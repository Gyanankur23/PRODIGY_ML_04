import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from PIL import Image

# Page config
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🖐️", layout="centered")

# Load model and class labels
MODEL_PATH = 'hand_gesture_model.h5'
CLASS_LABELS_PATH = 'class_labels.npy'

# Gesture classes
GESTURES = ['call_me', 'fingers_crossed', 'okay', 'paper', 'peace', 
            'rock', 'rock_on', 'scissor', 'thumbs', 'up']

@st.cache_resource
def load_model_and_labels():
    """Load the trained model and class labels"""
    model = None
    class_labels = {}
    
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully!")
    else:
        st.warning("Model file not found. Please train the model first using train_model.py")
    
    if os.path.exists(CLASS_LABELS_PATH):
        class_labels = np.load(CLASS_LABELS_PATH, allow_pickle=True).item()
    else:
        # Use default labels if file doesn't exist
        class_labels = {i: gesture for i, gesture in enumerate(GESTURES)}
    
    return model, class_labels

def predict_gesture(img_array, model, class_labels):
    """Predict hand gesture from image array"""
    if model is None:
        return None, 0
    
    try:
        # Preprocess image
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Get gesture name
        gesture_name = class_labels.get(predicted_class, GESTURES[predicted_class])
        
        return gesture_name, confidence
    except Exception as e:
        st.error(f"Error predicting gesture: {str(e)}")
        return None, 0

# Main app
def main():
    st.title("🖐️ Hand Gesture Recognition")
    st.markdown("---")
    
    # Load model
    model, class_labels = load_model_and_labels()
    
    # File upload
    uploaded_file = st.file_uploader("Upload an image of a hand gesture", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to array for prediction
        img_array = image.img_to_array(pil_image.resize((128, 128)))
        
        # Predict button
        if st.button("Predict Gesture", type="primary"):
            with st.spinner("Analyzing gesture..."):
                gesture_name, confidence = predict_gesture(img_array, model, class_labels)
                
                if gesture_name:
                    # Display results
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Detected Gesture",
                            gesture_name.replace('_', ' ').upper(),
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%",
                            delta=None
                        )
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Display all probabilities
                    st.markdown("---")
                    st.subheader("All Probabilities:")
                    
                    if model is not None:
                        img_array_norm = img_array / 255.0
                        img_array_norm = np.expand_dims(img_array_norm, axis=0)
                        predictions = model.predict(img_array_norm, verbose=0)[0]
                        
                        for i, (gesture, prob) in enumerate(zip(GESTURES, predictions)):
                            st.write(f"{gesture.replace('_', ' ').title()}: {prob * 100:.2f}%")
    
    # Instructions
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    1. Upload an image of a hand gesture
    2. Click "Predict Gesture" to analyze
    3. View the detected gesture and confidence level
    
    **Supported gestures:**
    - Call Me
    - Fingers Crossed
    - Okay
    - Paper
    - Peace
    - Rock
    - Rock On
    - Scissor
    - Thumbs Up
    - Up
    """)
    
    # Model info
    if model is not None:
        st.markdown("---")
        st.info(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        st.markdown("---")
        st.error(f"❌ Model not found. Run `python train_model.py` to train the model.")

if __name__ == '__main__':
    main()
