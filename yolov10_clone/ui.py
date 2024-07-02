from ultralytics import YOLOv10
from PIL import Image
import streamlit as st


def process_image(image):
    trained_model_path = "best.pt"
    no_doctor_image = Image.open("no_doctor_picture.jpg")

    model = YOLOv10(trained_model_path)
    conf_threshold = 0.3
    img_size = 640
    results = model.predict(source=image, conf=conf_threshold, imgsz=img_size)
    annotated_img = results[0].plot()
    image_rgb = annotated_img[..., ::-1]

    if results[0].boxes.conf.data.max().item() < 0.6:
        return no_doctor_image
    else:
        return image_rgb


def run():
    st.title("Doctor Uniform Detection for IMage")
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")
        image = Image.open(file)
        prc_image = process_image(image)
        st.image(prc_image, caption="Processed Image")


run()
