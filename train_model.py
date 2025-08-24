import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

# Path to images
train_images_path = r"D:\My Learning\External Work\Arabian Academy\Session 11\butterfly_classifier\train_images"

# Read CSV file
df = pd.read_csv("train.csv")

# Split the dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only preprocessing for validation set
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_images_path,
    x_col="filename",
    y_col="label",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    directory=train_images_path,
    x_col="filename",
    y_col="label",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

# Number of classes
num_classes = len(train_gen.class_indices)

# Load MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Phase 1: Freeze all layers

# Build the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Phase 1: Train Dense layers only
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Phase 1: Training Dense layers only...")
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Phase 2: Unfreeze all layers
base_model.trainable = True
for layer in base_model.layers:
    layer.trainable = True

# Re-compile with a very small learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Phase 2: Fine-tuning the entire model...")
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save the model
os.makedirs("model", exist_ok=True)
model.save("model/butterfly_model.keras")
print("Model saved in model/butterfly_model.keras")