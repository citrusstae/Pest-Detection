import tensorflow as tf
from tensorflow.keras import Sequential, layers
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras_tuner import  RandomSearch
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def build_model(hp):
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(hp.Int('conv1_units', min_value=32, max_value=128, step=32),
#                             (3, 3),
#                             activation='relu',
#                             input_shape=(120, 120, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(hp.Int('conv2_units', min_value=64, max_value=256, step=64),
#                             (3, 3),
#                             activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Dropout(0.3))
#     # model.add(layers.Conv2D(hp.Int('conv3_units', min_value=128, max_value=512, step=128),
#     #                         (3, 3),
#     #                         activation='relu'))
#     # model.add(layers.MaxPooling2D((2, 2)))
#     # model.add(layers.Dropout(0.2))
#     # model.add(layers.Conv2D(hp.Int('conv4_units', min_value=256, max_value=1024, step=256),
#     #                         (3, 3),
#     #                         activation='relu'))
#     # model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128),
#                            activation='relu'))
#     model.add(layers.Dropout(0.15))
#     model.add(layers.Dense(24, activation='softmax'))  # Assuming 24 classes
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


def create_and_compile_model():
    model = tf.keras.models.Sequential([
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(120,120,3)),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        # layers.Dropout(0.3),
        # layers.Conv2D(32, 3, activation='relu'),
        # layers.MaxPooling2D(pool_size=(2,2)),
        # layers.Dropout(0.3),
        # layers.Conv2D(256, 3, activation='relu'),
        # layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(24, activation='softmax')

    ])
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

# Call the function to create and compile the model
model = create_and_compile_model()

# Load and preprocess data
data_dir = Path(r'C:\Users\Lenovo\Desktop\24classes\fulldataset2')

# Define dictionary with class labels and file paths
dict = {
    "ants": list(data_dir.glob("ants*")),
    "bees": list(data_dir.glob("bees*")),
    "beetle": list(data_dir.glob("beetle*")),
    "catterpillar": list(data_dir.glob("catterpillar*")),
    "earthworms": list(data_dir.glob("earthworms*")),
    "earwig": list(data_dir.glob("earwig*")),
    "grasshopper": list(data_dir.glob("grasshopper*")),
    "moth": list(data_dir.glob("moth*")),
    "slug": list(data_dir.glob("slug*")),
    "snail": list(data_dir.glob("snail*")),
    "wasp": list(data_dir.glob("wasp*")),
    "Weevil": list(data_dir.glob("Weevil*")),
    "aphids": list(data_dir.glob("aphids*")),
    "armyworms": list(data_dir.glob("armyworms*")),
    "cabbageloopers": list(data_dir.glob("cabbageloopers*")),
    "cornborers": list(data_dir.glob("cornborers*")),
    "earworms": list(data_dir.glob("earworms*")),
    "fruitflies": list(data_dir.glob("fruitflies*")),
    "hornworms": list(data_dir.glob("hornworms*")),
    "potatobeetles": list(data_dir.glob("potatobeetles*")),
    "rootworms": list(data_dir.glob("rootworms*")),
    "spidermites": list(data_dir.glob("spidermites*")),
    "stinkbugs": list(data_dir.glob("stinkbugs*")),
    "thrips": list(data_dir.glob("thrips*"))
}

# Convert WindowsPath objects to strings in the dictionary values
for key, value in dict.items():
    dict[key] = [str(path) for path in value]

# Define labels dictionary
labels_dict = {
    'ants': 0, 'aphids': 1, 'armyworms': 2, 'bees': 3, 'beetle': 4,
    'cabbageloopers': 5, 'catterpillar': 6, 'cornborers': 7, 'earthworms': 8, 'fruitflies': 9,
    'grasshopper': 10, 'hornworms': 11, 'moth': 12, 'potatobeetles': 13, 'rootworms': 14, 'slug': 15,
    'snail': 16, 'spidermites': 17, 'stinkbugs': 18, 'thrips': 19, 'wasp': 20,
    'Weevil': 21, 'earwig': 22, 'earworms' : 23
}

# Define augmentation pipeline
aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=0.7),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# Load and preprocess images
X, y = [], []
for name, images in dict.items():
    for image in images:
        img = cv2.imread(image)
        resized_img = cv2.resize(img, (120, 120))  # Resize images
        X.append(resized_img)
        y.append(labels_dict[name])

X = np.array(X)
y = np.array(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=0)

# Define batch size
batch_size = 64


# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(labels_dict))
y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(labels_dict))

# Create data generators for training and validation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)    
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

# Define a tuner
tuner = RandomSearch(
    create_and_compile_model,
    objective='val_accuracy',
    max_trials=5,  # Number of hyperparameter combinations to try
    directory='random_search',
    project_name='cnn_hyperparameters'
)

# Perform the search
tuner.search(train_generator, epochs=50, validation_data=val_generator)

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]

best_hyperparameters = tuner.get_best_hyperparameters()[0]

# Print the hyperparameters
print("Best Hyperparameters:")
print(best_hyperparameters)


# Train the best model on the entire training data
history= model.fit(train_generator, epochs=10
                   , validation_data=val_generator)
# Print model summary
model.summary()
#plot training and validation accuracy 
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
model.save(r"C:\Users\Lenovo\Desktop\24classes\full_model.h5")

# X_test = X_val  # Assuming you want to evaluate on the validation set
# y_test = y_val  # Assuming the corresponding labels are also validation labels

# # Make predictions
# y_pred_prob = model.predict(X_test)
# y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

# # Calculate metrics
# accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
# precision = precision_score(np.argmax(y_test, axis=1), y_pred, average='macro')
# recall = recall_score(np.argmax(y_test, axis=1), y_pred, average='macro')
# f1 = f1_score(np.argmax(y_test, axis=1), y_pred, average='macro')

# # Print the metrics
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1-score: {f1}')
