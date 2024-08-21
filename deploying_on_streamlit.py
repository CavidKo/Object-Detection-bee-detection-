import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

SAVE_DIR = 'test/result/'

# Load the YOLO model
model = YOLO('beeModel2.pt')

# Streamlit app
st.title("Object Detection with YOLO")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Run YOLO model on the uploaded image
    with st.spinner('Processing...'):
        results = model(source=image, conf=0.3)

    # Display the results
    result_image = results[0].plot()
    result_pil_image = Image.fromarray(result_image)

    # Display the results
    st.image(result_image, caption='Detected Objects', use_column_width=True)

    # Save the result image if needed
    result_pil_image.save(f'{SAVE_DIR}/output.jpg')  # Save the image using PIL's save method

    st.success('Detection complete!')

