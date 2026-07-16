import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set paths
DATASET_PATH = 'HandGesture/images'
MODEL_PATH = 'hand_gesture_model.h5'

# Gesture classes
GESTURES = ['call_me', 'fingers_crossed', 'okay', 'paper', 'peace', 
            'rock', 'rock_on', 'scissor', 'thumbs', 'up']

# Image parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 50

def create_model():
    """Create CNN model for hand gesture recognition"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(GESTURES), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model():
    """Train the hand gesture recognition model"""
    print("Setting up data generators...")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Load and prepare training data
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        classes=GESTURES
    )
    
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        classes=GESTURES
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    
    # Create model
    model = create_model()
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy')
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Save final model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save class labels
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    np.save('class_labels.npy', class_labels)
    print("Class labels saved to class_labels.npy")
    
    return history, model

if __name__ == '__main__':
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} not found!")
        print("Please ensure the HandGesture/images directory exists with gesture subdirectories.")
        exit(1)
    
    history, model = train_model()
    print("Training completed successfully!")
