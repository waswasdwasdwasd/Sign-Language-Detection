# %% Cell 1: Importing Libraries
import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from collections import deque

print("All libraries imported successfully!")

# %% Cell 2: 
# DATA COLLECTION WITH AUTO-ADVANCE
# Press SPACE to move to next letter, 's' to save sample, 'q' to quit anytime

# Define all letters you want to collect (modify as needed)
ALL_GESTURES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

SAMPLES_PER_GESTURE = 300  # Collect 300 samples per gesture for better accuracy

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Support two hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create data directory
os.makedirs('gesture_data', exist_ok=True)

cap = cv2.VideoCapture(0)
current_gesture_idx = 0
current_gesture = ALL_GESTURES[current_gesture_idx]
data_buffer = []

print(f"Starting data collection for {len(ALL_GESTURES)} gestures")
print("Controls:")
print("  's' - Save current hand pose")
print("  SPACE - Move to next letter")
print("  'q' - Quit and save")
print(f"\nCollecting {SAMPLES_PER_GESTURE} samples per gesture")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Display info on screen
    info_text = f"Gesture: {current_gesture} ({current_gesture_idx+1}/{len(ALL_GESTURES)})"
    samples_text = f"Samples: {len(data_buffer)}/{SAMPLES_PER_GESTURE}"
    
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, samples_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, "Press 's' to save | SPACE for next | 'q' to quit", 
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Process hands
    if results.multi_hand_landmarks:
        all_landmarks = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks for this hand
            hand_coords = []
            for landmark in hand_landmarks.landmark:
                hand_coords.extend([landmark.x, landmark.y, landmark.z])
            all_landmarks.extend(hand_coords)
        
        # Pad if only one hand detected (ensure consistent 126-dimensional input)
        while len(all_landmarks) < 126:  # 2 hands × 21 landmarks × 3 coords
            all_landmarks.extend([0.0] * 63)
        
        # Save sample when 's' pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            data_buffer.append(all_landmarks[:126])  # Ensure exactly 126 features
            print(f"✓ Saved sample {len(data_buffer)} for '{current_gesture}'")
            
        elif key == ord(' '):  # Spacebar pressed
            # Save current gesture data
            if len(data_buffer) > 0:
                np.save(f'gesture_data/gesture_{current_gesture}.npy', np.array(data_buffer))
                print(f"Saved {len(data_buffer)} samples for '{current_gesture}'")
            
            # Move to next gesture
            current_gesture_idx += 1
            if current_gesture_idx >= len(ALL_GESTURES):
                print("All gestures collected!")
                break
            
            current_gesture = ALL_GESTURES[current_gesture_idx]
            data_buffer = []
            print(f"\n--- Now collecting: {current_gesture} ---")
            
        elif key == ord('q'):
            # Save and quit
            if len(data_buffer) > 0:
                np.save(f'gesture_data/gesture_{current_gesture}.npy', np.array(data_buffer))
                print(f"Saved {len(data_buffer)} samples for '{current_gesture}'")
            break
    else:
        cv2.putText(frame, "No hand detected", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.imshow('Data Collection', frame)

cap.release()
cv2.destroyAllWindows()
print("\nData collection complete!")

# %% Cell 3:
# DATA AUGMENTATION for improved accuracy

def augment_landmarks(landmarks, augmentation_type='rotation', param=None):
    """
    Apply data augmentation to hand landmarks
    landmarks: array of shape (126,) for two hands or (63,) for one hand
    """
    landmarks = landmarks.reshape(-1, 3)  # Reshape to (N, 3) where N=21 or 42
    
    if augmentation_type == 'rotation':
        # Rotate around z-axis (perpendicular to camera)
        angle = param if param else np.random.uniform(-30, 30)  # degrees
        angle_rad = np.radians(angle)
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        landmarks = landmarks @ rotation_matrix.T
        
    elif augmentation_type == 'scale':
        # Scale hand size
        scale_factor = param if param else np.random.uniform(0.8, 1.2)
        landmarks = landmarks * scale_factor
        
    elif augmentation_type == 'translate':
        # Random translation
        translation = param if param else np.random.uniform(-0.1, 0.1, (1, 3))
        landmarks = landmarks + translation
        
    elif augmentation_type == 'noise':
        # Add random noise
        noise_level = param if param else 0.02
        noise = np.random.normal(0, noise_level, landmarks.shape)
        landmarks = landmarks + noise
        
    elif augmentation_type == 'flip':
        # Horizontal flip (mirror)
        landmarks[:, 0] = 1.0 - landmarks[:, 0]
    
    return landmarks.flatten()

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist position"""
    landmarks = landmarks.reshape(-1, 3)
    
    # Normalize each hand separately
    num_hands = len(landmarks) // 21
    normalized = []
    
    for i in range(num_hands):
        hand_landmarks = landmarks[i*21:(i+1)*21]
        wrist = hand_landmarks[0]
        hand_landmarks = hand_landmarks - wrist
        
        # Scale by max distance
        max_dist = np.max(np.abs(hand_landmarks)) + 1e-8
        hand_landmarks = hand_landmarks / max_dist
        normalized.append(hand_landmarks)
    
    if len(normalized) == 0:
        return landmarks.flatten()
    
    return np.concatenate(normalized).flatten()

print("Data augmentation functions ready!")

# %% Cell 4:
# LOAD DATA WITH AUGMENTATION

# Load all gesture files
gesture_files = [f for f in os.listdir('gesture_data') if f.endswith('.npy')]
gestures = sorted([f.replace('gesture_', '').replace('.npy', '') for f in gesture_files])

print(f"Found {len(gestures)} gestures: {gestures}")

X = []
y = []

for idx, gesture in enumerate(gestures):
    filepath = f'gesture_data/gesture_{gesture}.npy'
    if os.path.exists(filepath):
        data = np.load(filepath)
        print(f"Loaded {len(data)} samples for '{gesture}'")
        
        # Add original samples
        for sample in data:
            normalized = normalize_landmarks(sample)
            X.append(normalized)
            y.append(idx)
        
        # Add augmented samples (2x data through augmentation)
        print(f"  Augmenting data for '{gesture}'...")
        for sample in data[:len(data)//2]:  # Augment half the dataset
            # Rotation augmentation
            aug_sample = augment_landmarks(sample, 'rotation', np.random.uniform(-25, 25))
            X.append(normalize_landmarks(aug_sample))
            y.append(idx)
            
            # Scale augmentation
            aug_sample = augment_landmarks(sample, 'scale', np.random.uniform(0.85, 1.15))
            X.append(normalize_landmarks(aug_sample))
            y.append(idx)
            
            # Noise augmentation
            aug_sample = augment_landmarks(sample, 'noise', 0.015)
            X.append(normalize_landmarks(aug_sample))
            y.append(idx)

X = np.array(X)
y = np.array(y)

print(f"\nTotal dataset size: {len(X)} samples")
print(f"Feature dimensions: {X.shape[1]}")
print(f"Number of classes: {len(gestures)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# %% Cell 5:
# BUILD IMPROVED MODEL with better architecture

num_classes = len(gestures)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Enhanced model architecture
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(32, activation='relu'),
    
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for better training
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_sign_model.h5', monitor='val_accuracy', 
                            save_best_only=True, mode='max', verbose=1)

print("Model architecture:")
model.summary()

# %% Cell 6:
# TRAIN MODEL
print("\nStarting training...")
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n{'='*50}")
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")
print(f"{'='*50}")

# Save final model
model.save('sign_language_model.h5')
np.save('gesture_labels.npy', gestures)
print("Model saved as 'sign_language_model.h5'")

# %% Cell 7:
# BUILD LSTM MODEL for word/sequence recognition

SEQUENCE_LENGTH = 30  # Number of frames to capture per gesture

# Prepare sequence data (reshape for LSTM)
# For LSTM, we need shape: (samples, timesteps, features)
# We'll create this during real-time prediction

def create_lstm_model(input_shape, num_classes):
    """Create LSTM model for temporal gesture recognition"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create LSTM model (will be used in real-time recognition)
lstm_model = create_lstm_model((SEQUENCE_LENGTH, X_train.shape[1]), num_classes)
print("LSTM model created for sequence recognition")
lstm_model.summary()

# Note: To fully train LSTM, you'd need to collect temporal sequences
# For now, we'll use the static model with temporal buffering in real-time

# %% Cell 8:
# REAL-TIME RECOGNITION WITH WORD FORMATION

# Load model and labels
model = load_model('sign_language_model.h5')
gestures = np.load('gesture_labels.npy')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Word formation variables
detected_letters = []
current_word = ""
last_detection_time = time.time()
LETTER_CONFIRMATION_TIME = 1.5  # Seconds to hold gesture to confirm letter
WORD_PAUSE_TIME = 3.0  # Seconds of no detection to finalize word
last_predicted_gesture = None
gesture_start_time = None
confidence_history = deque(maxlen=10)  # Smooth predictions

print("Real-time recognition started!")
print("Controls:")
print("  'c' - Clear current word")
print("  'q' - Quit")
print("\nHold gesture for 1.5 seconds to add letter")
print("Pause for 3 seconds to complete word")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    current_time = time.time()
    predicted_gesture = None
    confidence = 0
    
    if results.multi_hand_landmarks:
        all_landmarks = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_coords = []
            for landmark in hand_landmarks.landmark:
                hand_coords.extend([landmark.x, landmark.y, landmark.z])
            all_landmarks.extend(hand_coords)
        
        # Pad to 126 dimensions
        while len(all_landmarks) < 126:
            all_landmarks.extend([0.0] * 63)
        
        # Normalize and predict
        landmarks_normalized = normalize_landmarks(np.array(all_landmarks[:126]))
        prediction = model.predict(landmarks_normalized.reshape(1, -1), verbose=0)
        gesture_idx = np.argmax(prediction)
        confidence = prediction[0][gesture_idx]
        
        # Smooth predictions with confidence history
        confidence_history.append((gesture_idx, confidence))
        
        if confidence > 0.75:  # High confidence threshold
            predicted_gesture = gestures[gesture_idx]
            
            # Letter confirmation logic
            if predicted_gesture == last_predicted_gesture:
                if gesture_start_time is None:
                    gesture_start_time = current_time
                elif current_time - gesture_start_time >= LETTER_CONFIRMATION_TIME:
                    # Letter confirmed - add to word
                    if len(detected_letters) == 0 or detected_letters[-1] != predicted_gesture:
                        detected_letters.append(predicted_gesture)
                        current_word += predicted_gesture
                        print(f"Letter added: {predicted_gesture} | Word: {current_word}")
                    gesture_start_time = None  # Reset
            else:
                last_predicted_gesture = predicted_gesture
                gesture_start_time = current_time
            
            last_detection_time = current_time
            
            # Draw prediction with hold progress
            hold_time = 0
            if gesture_start_time:
                hold_time = current_time - gesture_start_time
            
            progress = min(hold_time / LETTER_CONFIRMATION_TIME, 1.0)
            progress_bar_width = int(progress * 200)
            
            cv2.rectangle(frame, (10, 120), (210, 150), (100, 100, 100), -1)
            cv2.rectangle(frame, (10, 120), (10 + progress_bar_width, 150), (0, 255, 0), -1)
            cv2.putText(frame, f'{predicted_gesture} ({confidence:.2f})', 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    else:
        # No hand detected - check for word pause
        if current_time - last_detection_time > WORD_PAUSE_TIME and current_word:
            print(f"\n*** WORD COMPLETED: {current_word} ***\n")
            detected_letters = []
            current_word = ""
        
        gesture_start_time = None
        last_predicted_gesture = None
    
    # Display current word
    cv2.putText(frame, f'Word: {current_word}', 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    
    # Display instructions
    cv2.putText(frame, "Hold gesture 1.5s to add letter | Pause 3s to finish word", 
               (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Sign Language Recognition', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_word = ""
        detected_letters = []
        print("Word cleared")

cap.release()
cv2.destroyAllWindows()
print("Recognition stopped")

# %% Cell 9:
# OPTIONAL: Real-time with LSTM sequence buffer
# This version captures temporal sequences for more complex gestures

# This would require collecting sequential data first
# For now, use Cell 9 for word formation with static gestures

SEQUENCE_BUFFER = deque(maxlen=SEQUENCE_LENGTH)

def predict_with_lstm(sequence_buffer, lstm_model, gestures):
    """Predict gesture from sequence of frames"""
    if len(sequence_buffer) < SEQUENCE_LENGTH:
        return None, 0
    
    sequence = np.array(list(sequence_buffer))
    sequence = sequence.reshape(1, SEQUENCE_LENGTH, -1)
    
    prediction = lstm_model.predict(sequence, verbose=0)
    gesture_idx = np.argmax(prediction)
    confidence = prediction[0][gesture_idx]
    
    return gestures[gesture_idx], confidence

print("LSTM sequence recognition function ready")
print("To use this, collect temporal sequence data and train the LSTM model")
