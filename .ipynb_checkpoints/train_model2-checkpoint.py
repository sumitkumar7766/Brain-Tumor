import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# 1. Dataset Paths
base_path = '/kaggle/input/datasets/sartajbhuvaji/brain-tumor-classification-mri'
train_dir = os.path.join(base_path, 'Training')
test_dir = os.path.join(base_path, 'Testing')

if not os.path.exists(train_dir):
    print("Path nahi mila! Kripya Kaggle me dataset add karke path check karein.")
else:
    print("Dataset mil gaya! Anti-Overfitting Training shuru karte hain...")

# 2. Data Preprocessing & Augmentation (Overfitting Rokne Ke Liye Best Settings)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# TRICK 1: rescale ki jagah official preprocess_input use karein
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1, # Naya: thoda zoom bhi add kiya hai
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

print("\nLoading Training Data:")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("\nLoading Testing Data:")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False 
)

# 3. Build Transfer Learning Model (MobileNetV2)
with tf.device('/GPU:0'): 
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Shuru me freeze rakhenge

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        # TRICK 2: L2 Regularization add kiya hai overfitting rokne ke liye
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)), 
        Dropout(0.5),
        Dense(4, activation='softmax') 
    ])

    # TRICK 3: Learning rate thoda kam kiya (0.001 se 0.0005) stability ke liye
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# 4. Callbacks (Smart Training)
callbacks = [
    # Best model save karega
    ModelCheckpoint('tumor_classifier_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    
    # Patience thodi badha di hai (7) taaki ReduceLR apna kaam kar sake
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    
    # TRICK 4: Agar 2 epoch tak val_loss improve nahi hua, toh learning rate aadhi ho jayegi
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

# 5. Train the Model
print("\nStarting Model Training...")
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30, # Epochs 30 kar diye hain kyunki early stopping apne aap rok dega agar zaroorat padi
    callbacks=callbacks
)

# 6. Save Final Model
model.save('tumor_classifier_final.keras')
print("\nTraining Complete! Model saved as 'tumor_classifier_final.keras'")