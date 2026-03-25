"""
Complete pipeline for collecting sign language data and training improved models
Run this to build your own custom dataset and train models
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path
import pickle
import json
from collections import defaultdict
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt


class DataCollector:
    """Collect sign language gesture data"""
    
    def __init__(self, data_dir: str = "MP_Data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.sequence_length = 30
        self.num_sequences = 30  # Sequences per sign
        
    def extract_keypoints(self, results):
        """Extract hand keypoints from MediaPipe results"""
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            # 21 landmarks * 3 coordinates = 63 features
            keypoints = []
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return np.array(keypoints)
        return np.zeros(63)
    
    def collect_for_sign(self, sign_name: str, sign_index: int):
        """Collect data for a specific sign"""
        print(f"\n📹 Collecting data for sign: {sign_name}")
        print(f"You'll record {self.num_sequences} sequences")
        print(f"Each sequence = {self.sequence_length} frames")
        
        # Create directory for this sign
        sign_dir = self.data_dir / sign_name
        sign_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        
        for sequence in range(self.num_sequences):
            sequence_data = []
            
            print(f"\n🎬 Recording sequence {sequence + 1}/{self.num_sequences}")
            print("Press 'r' when ready...")
            
            # Wait for user to be ready
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                
                cv2.putText(frame, f"Sign: {sign_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Sequence: {sequence + 1}/{self.num_sequences}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'r' to start recording", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(10) & 0xFF
                if key == ord('r'):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            # Collect frames
            print("🔴 RECORDING...")
            for frame_num in range(self.sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                        )
                
                # Extract keypoints
                keypoints = self.extract_keypoints(results)
                sequence_data.append(keypoints)
                
                # Display progress
                progress = int((frame_num / self.sequence_length) * 100)
                cv2.rectangle(frame, (10, 150), (10 + progress * 3, 170), (0, 255, 0), -1)
                cv2.putText(frame, f"Recording: {frame_num + 1}/{self.sequence_length}", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Data Collection', frame)
                cv2.waitKey(10)
            
            # Save sequence
            sequence_array = np.array(sequence_data)
            save_path = sign_dir / f"seq_{sequence}.npy"
            np.save(save_path, sequence_array)
            print(f"✅ Saved sequence {sequence + 1}")
            
            time.sleep(0.5)  # Brief pause between sequences
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def collect_dataset(self, signs: list):
        """Collect data for multiple signs"""
        print("=" * 60)
        print("🎯 SIGN LANGUAGE DATA COLLECTION")
        print("=" * 60)
        print(f"\nYou will collect data for {len(signs)} signs:")
        for i, sign in enumerate(signs, 1):
            print(f"  {i}. {sign}")
        
        print("\n📋 Instructions:")
        print("  - Position yourself in good lighting")
        print("  - Keep hand clearly visible")
        print("  - Perform gesture naturally")
        print("  - Press 'r' to start each sequence")
        print("  - Press 'q' to quit anytime")
        
        input("\nPress Enter to start...")
        
        for idx, sign in enumerate(signs):
            if not self.collect_for_sign(sign, idx):
                print("\n⚠️ Collection stopped by user")
                break
        
        print("\n✅ Data collection complete!")
        print(f"📁 Data saved to: {self.data_dir}")


class DataAugmenter:
    """Augment training data for better generalization"""
    
    @staticmethod
    def add_noise(sequence, noise_factor=0.01):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_factor, sequence.shape)
        return sequence + noise
    
    @staticmethod
    def scale_sequence(sequence, scale_range=(0.9, 1.1)):
        """Scale the gesture"""
        scale = np.random.uniform(*scale_range)
        return sequence * scale
    
    @staticmethod
    def rotate_sequence(sequence, max_angle=15):
        """Rotate gesture in 2D plane"""
        angle = np.random.uniform(-max_angle, max_angle)
        angle_rad = np.radians(angle)
        
        # Reshape to (frames, 21, 3)
        frames, features = sequence.shape
        landmarks = sequence.reshape(frames, 21, 3)
        
        # Rotation matrix for x-y plane
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to each frame
        rotated = np.zeros_like(landmarks)
        for i in range(frames):
            rotated[i] = landmarks[i] @ rotation_matrix.T
        
        return rotated.reshape(frames, features)
    
    @staticmethod
    def time_warp(sequence, warp_factor=0.2):
        """Time warping - speed up or slow down"""
        frames = len(sequence)
        new_length = int(frames * np.random.uniform(1 - warp_factor, 1 + warp_factor))
        
        # Interpolate to new length
        indices = np.linspace(0, frames - 1, new_length)
        warped = np.zeros((new_length, sequence.shape[1]))
        
        for feature_idx in range(sequence.shape[1]):
            warped[:, feature_idx] = np.interp(indices, np.arange(frames), sequence[:, feature_idx])
        
        # Resize back to original length
        final_indices = np.linspace(0, new_length - 1, frames)
        final = np.zeros_like(sequence)
        for feature_idx in range(sequence.shape[1]):
            final[:, feature_idx] = np.interp(final_indices, np.arange(new_length), warped[:, feature_idx])
        
        return final
    
    @staticmethod
    def augment_batch(sequences, labels, augment_factor=2):
        """Augment entire batch"""
        augmented_sequences = []
        augmented_labels = []
        
        for seq, label in zip(sequences, labels):
            # Original
            augmented_sequences.append(seq)
            augmented_labels.append(label)
            
            # Augmentations
            for _ in range(augment_factor):
                aug_seq = seq.copy()
                
                # Apply random augmentations
                if np.random.rand() > 0.5:
                    aug_seq = DataAugmenter.add_noise(aug_seq)
                if np.random.rand() > 0.5:
                    aug_seq = DataAugmenter.scale_sequence(aug_seq)
                if np.random.rand() > 0.5:
                    aug_seq = DataAugmenter.rotate_sequence(aug_seq)
                if np.random.rand() > 0.5:
                    aug_seq = DataAugmenter.time_warp(aug_seq)
                
                augmented_sequences.append(aug_seq)
                augmented_labels.append(label)
        
        return np.array(augmented_sequences), np.array(augmented_labels)


class ImprovedModelTrainer:
    """Train improved models with attention mechanisms"""
    
    def __init__(self, data_dir: str = "MP_Data"):
        self.data_dir = Path(data_dir)
        self.sequence_length = 30
        self.num_features = 63
        
    def load_data(self):
        """Load collected data"""
        sequences = []
        labels = []
        label_map = {}
        
        sign_folders = [f for f in self.data_dir.iterdir() if f.is_dir()]
        
        for sign_idx, sign_folder in enumerate(sorted(sign_folders)):
            sign_name = sign_folder.name
            label_map[sign_idx] = sign_name
            
            # Load all sequences for this sign
            for seq_file in sign_folder.glob("seq_*.npy"):
                sequence = np.load(seq_file)
                
                # Ensure correct shape
                if sequence.shape[0] == self.sequence_length and sequence.shape[1] == self.num_features:
                    sequences.append(sequence)
                    labels.append(sign_idx)
        
        print(f"✅ Loaded {len(sequences)} sequences for {len(label_map)} signs")
        print(f"📋 Signs: {list(label_map.values())}")
        
        return np.array(sequences), np.array(labels), label_map
    
    def build_lstm_attention_model(self, num_classes):
        """Build LSTM with attention mechanism"""
        inputs = layers.Input(shape=(self.sequence_length, self.num_features))
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
        x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_cnn_lstm_model(self, num_classes):
        """Build CNN-LSTM hybrid model"""
        inputs = layers.Input(shape=(self.sequence_length, self.num_features))
        
        # Reshape for 1D convolution
        x = layers.Reshape((self.sequence_length, self.num_features, 1))(inputs)
        
        # CNN layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Flatten for LSTM
        x = layers.Reshape((-1, 64 * self.num_features))(x)
        
        # LSTM layers
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def train(self, model_type='lstm_attention', augment=True, epochs=100):
        """Train the model"""
        print("\n" + "=" * 60)
        print(f"🚀 TRAINING {model_type.upper()} MODEL")
        print("=" * 60)
        
        # Load data
        X, y, label_map = self.load_data()
        num_classes = len(label_map)
        
        # Data augmentation
        if augment:
            print("\n🔄 Applying data augmentation...")
            X, y = DataAugmenter.augment_batch(X, y, augment_factor=3)
            print(f"✅ Augmented to {len(X)} sequences")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Dataset split:")
        print(f"  Training: {len(X_train)} sequences")
        print(f"  Testing: {len(X_test)} sequences")
        
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes)
        
        # Build model
        print(f"\n🏗️ Building {model_type} model...")
        if model_type == 'lstm_attention':
            model = self.build_lstm_attention_model(num_classes)
        elif model_type == 'cnn_lstm':
            model = self.build_cnn_lstm_model(num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        
        # Callbacks
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(model_dir / f"{model_type}_best.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger(
                filename=str(model_dir / f"{model_type}_training.log")
            )
        ]
        
        # Train
        print("\n🏋️ Training started...")
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=epochs,
            batch_size=32,
            callbacks=callback_list,
            verbose=1
        )
        
        # Evaluate
        print("\n📈 Final evaluation:")
        test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        # Save model and labels
        final_model_path = model_dir / f"{model_type}_final.h5"
        model.save(final_model_path)
        print(f"\n💾 Model saved to: {final_model_path}")
        
        labels_path = self.data_dir / "labels.npy"
        np.save(labels_path, list(label_map.values()))
        print(f"💾 Labels saved to: {labels_path}")
        
        # Plot training history
        self._plot_history(history, model_type)
        
        return model, history, label_map
    
    def _plot_history(self, history, model_type):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title(f'{model_type} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title(f'{model_type} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'models/{model_type}_training_history.png', dpi=150)
        print(f"📊 Training plot saved to: models/{model_type}_training_history.png")
        plt.close()


def main():
    """Main execution flow"""
    print("\n" + "=" * 70)
    print("🎯 SIGN LANGUAGE RECOGNITION - DATA COLLECTION & TRAINING PIPELINE")
    print("=" * 70)
    
    print("\nChoose an option:")
    print("  1. Collect new data")
    print("  2. Train model on existing data")
    print("  3. Collect data AND train model")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    # Define signs to collect
    signs = [
        "hello", "thanks", "please", "sorry", "yes", 
        "no", "help", "stop", "iloveyou", "goodbye"
    ]
    
    if choice in ['1', '3']:
        # Collect data
        collector = DataCollector(data_dir="MP_Data")
        collector.collect_dataset(signs)
    
    if choice in ['2', '3']:
        # Train model
        trainer = ImprovedModelTrainer(data_dir="MP_Data")
        
        print("\nChoose model architecture:")
        print("  1. LSTM with Attention (Recommended)")
        print("  2. CNN-LSTM Hybrid")
        
        model_choice = input("\nEnter choice (1-2): ").strip()
        model_type = 'lstm_attention' if model_choice == '1' else 'cnn_lstm'
        
        model, history, label_map = trainer.train(
            model_type=model_type,
            augment=True,
            epochs=100
        )
        
        print("\n✅ Training complete!")
        print(f"🎯 Trained on signs: {list(label_map.values())}")
    
    print("\n" + "=" * 70)
    print("✨ ALL DONE! You can now use your trained model in the app.")
    print("=" * 70)


if __name__ == "__main__":
    main()