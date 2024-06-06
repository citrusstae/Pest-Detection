import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Load the saved model
path = 'pest_Recog_Model3.keras'
model = tf.keras.models.load_model(path)

# Define the class names
pest_names = ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
              'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']

# Define the classify_images function
def classify_images(image_data):
    input_image = Image.open(image_data).convert("RGB")
    input_image = input_image.resize((180, 180))  # Resize the image
    input_image_array = image.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_class_index = np.argmax(result)
    predicted_class_name = pest_names[predicted_class_index]
    predicted_score = np.max(result) * 100
    return predicted_class_name, predicted_score


# Streamlit app
st.title("FarmGuard: A Pest Detector")
st.write("FarmGuard is an innovative tool designed to assist farmers in identifying pests from images of their crops. By leveraging advanced image processing and machine learning techniques, the app can recognize and classify 12 different classes of pests, including ants, bees, beetles, and caterpillars. Users simply upload an image of the affected area, and the system processes the image to detect and annotate pest regions.")
video_file = open('farm.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)
# File upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# If image is uploaded
if uploaded_image is not None:
    # Display uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform prediction when button is clicked
    # Perform prediction when button is clicked
    if st.button("Detect"):
        # Call your model function to get text prediction
        predicted_class_name, predicted_score = classify_images(uploaded_image)

        # Display prediction
        st.write("Detected Pest:", predicted_class_name)
        st.write("Confidence Score:", predicted_score)




