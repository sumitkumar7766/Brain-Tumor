import os
import cv2
import numpy as np
import nibabel as nib
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Dataset Paths
try:
    TRAIN_DIR = glob.glob('/kaggle/input/**/MICCAI_BraTS2020_TrainingData', recursive=True)[0]
    VAL_DIR = glob.glob('/kaggle/input/**/MICCAI_BraTS2020_ValidationData', recursive=True)[0]
    print(f"Training Folder mil gaya: {TRAIN_DIR}")
    print(f"Validation Folder mil gaya: {VAL_DIR}")
except IndexError:
    raise FileNotFoundError("Dataset paths nahi mile. Kripya check karein.")

# --- PART A: LOAD TRAINING DATA (With Masks) ---
def load_training_data(base_dir, img_size=128):
    X, y = [], []
    patient_folders = sorted([f for f in os.listdir(base_dir) if f.startswith('BraTS20_Training_')])
    print(f"\nLoading Training Data from {len(patient_folders)} patients. Kripya wait karein...")

    for folder in patient_folders:
        patient_path = os.path.join(base_dir, folder)
        try:
            flair = nib.load(os.path.join(patient_path, f'{folder}_flair.nii')).get_fdata()
            t1ce  = nib.load(os.path.join(patient_path, f'{folder}_t1ce.nii')).get_fdata()
            t2    = nib.load(os.path.join(patient_path, f'{folder}_t2.nii')).get_fdata()
            mask  = nib.load(os.path.join(patient_path, f'{folder}_seg.nii')).get_fdata()
        except:
            continue

        flair = flair / np.max(flair)
        t1ce = t1ce / np.max(t1ce)
        t2 = t2 / np.max(t2)

        # Slices extract karein
        for i in range(40, 120): 
            slice_mask = mask[:, :, i]
            has_tumor = 1 if np.max(slice_mask) > 0 else 0
            
            combined_slice = np.stack([flair[:,:,i], t1ce[:,:,i], t2[:,:,i]], axis=-1)
            combined_slice_resized = cv2.resize(combined_slice, (img_size, img_size))
            
            X.append(combined_slice_resized.astype(np.float32))
            y.append(has_tumor)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int8)

# Load and Split Training Data internally for validation during training
X_data, y_labels = load_training_data(TRAIN_DIR)
X_train, X_val, y_train, y_val = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

print(f"Training Slices: {len(X_train)}, Validation Slices: {len(X_val)}")

# --- PART B: MODEL TRAINING ---
with tf.device('/GPU:0'): 
    model = Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('brain_tumor_detector_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

print("\nStarting Model Training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64, 
    epochs=30,
    callbacks=callbacks
)

# --- PART C: PREDICT ON VALIDATION FOLDER (No Masks) ---
def load_validation_data_for_prediction(base_dir, num_patients=5, img_size=128):
    """
    Ye function sirf X (images) load karega ValidationData folder se, kyunki y (masks) nahi hain.
    """
    X_unseen = []
    patient_folders = sorted([f for f in os.listdir(base_dir) if f.startswith('BraTS20_Validation_')])[:num_patients]
    print(f"\nLoading Validation Folder images for PREDICTION from {len(patient_folders)} patients...")

    for folder in patient_folders:
        patient_path = os.path.join(base_dir, folder)
        try:
            flair = nib.load(os.path.join(patient_path, f'{folder}_flair.nii')).get_fdata()
            t1ce  = nib.load(os.path.join(patient_path, f'{folder}_t1ce.nii')).get_fdata()
            t2    = nib.load(os.path.join(patient_path, f'{folder}_t2.nii')).get_fdata()
        except:
            continue

        flair = flair / np.max(flair)
        t1ce = t1ce / np.max(t1ce)
        t2 = t2 / np.max(t2)

        for i in range(70, 80): # Sirf 10 slices sample ke taur par utha rahe hain predict karne ke liye
            combined_slice = np.stack([flair[:,:,i], t1ce[:,:,i], t2[:,:,i]], axis=-1)
            combined_slice_resized = cv2.resize(combined_slice, (img_size, img_size))
            X_unseen.append(combined_slice_resized.astype(np.float32))

    return np.array(X_unseen, dtype=np.float32)

# Unseen Data (Validation Folder) load karna
X_unseen_data = load_validation_data_for_prediction(VAL_DIR)

print("\nPredicting on Validation Folder images (Bina answers ke):")
predictions = model.predict(X_unseen_data)
predicted_classes = (predictions > 0.5).astype(int).flatten()

print("\nSample Predictions for Unseen Data (Validation Folder):")
for i, pred in enumerate(predicted_classes[:20]): # Peli 20 predictions print kar rahe hain
    label = "Tumor Detected" if pred == 1 else "Healthy (No Tumor)"
    print(f"Slice {i+1}: {label} (Probability: {predictions[i][0]:.4f})")

print("\nTask Complete! Model properly trained and tested on Validation Folder.")