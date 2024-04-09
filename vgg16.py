import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt

# Define data directory
data_dir = Path(r"C:\Users\Lenovo\Desktop\24classes\organized12classes")

# Define image size and batch size
img_size = (120, 120)
batch_size = 16

# # Define augmentation pipeline
# def augment(image):
#     aug_pipeline = A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.RandomRotate90(p=0.7),
#         A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
#     ])
#     return aug_pipeline(image=image)['image']

# # Define data generator for training set with augmentation
# train_datagen = ImageDataGenerator(
#     preprocessing_function=augment,
#     validation_split=0.2
# )
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Splitting data into training and validation sets
)
# Define data generator for validation set
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load and prepare training data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Specify training subset
)

# Load and prepare validation data
val_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Specify validation subset
)

# Load the VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(120, 120, 3))

# Freeze the pre-trained layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Add custom fully connected layers on top of VGG16
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(12, activation='softmax')(x)  # Adjust output units according to your problem

# Create the model
model = Model(inputs=base_model.input, outputs=output)

# # Define the initial learning rate and decay schedule
# initial_learning_rate = 0.001
# lr_schedule = ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=1000,
#     decay_rate=0.9,
#     staircase=True
# )

# # Create the optimizer with the learning rate schedule
# optimizer = Adam(learning_rate=lr_schedule)
# Define the optimizer with a fixed learning rate
optimizer = Adam(learning_rate=0.001)

# Compile the model with custom learning rate and loss function
model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model using fit method
history = model.fit(train_generator, epochs=30, validation_data=val_generator)

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the best model
model.save(r'C:\Users\Lenovo\Desktop\24classes\organized12classes\pest_detect_model_vgg.h5')
model.summary()