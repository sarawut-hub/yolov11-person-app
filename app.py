import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.title("YOLO Image Detection License Plate App :)")

# Load YOLO model
model = YOLO("best.pt")

# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    # Show original image
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Read image and convert to numpy array
    image = Image.open(uploaded_image).convert("RGB")  # Ensure 3 channels
    image_np = np.array(image)

    # Convert RGB → BGR (OpenCV format) for YOLO
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO inference
    st.info("Running YOLO object detection...")
    results = model.predict(image_bgr, conf=0.4)

    # Draw results on image
    result_image = results[0].plot()
    st.image(result_image, caption="YOLO Detection Result", use_container_width=True)
    st.success("Detection completed!")

    # Extract detection results
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]

    # Count license plates
    license_count = class_names.count("licenseplate")
    st.write(f"Number of license plates detected: **{license_count}**")
