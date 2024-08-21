import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

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
    st.image(results[0].plot(), caption='Detected Objects', use_column_width=True)
    
    # Save the result image if needed
    # results.save(save_dir='output')

    st.success('Detection complete!')

