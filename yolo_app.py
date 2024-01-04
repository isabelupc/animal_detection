import streamlit as st
from PIL import Image
import torch
import numpy as np 
import cv2

# nomes per windows
#import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath
#######

@st.cache_resource
def load_model(path, device):
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    model_.to(device)
    print("model to ", device)
    return model_
 
def image_input():
    img_file = None

    img_bytes = st.file_uploader("Tria o arrosega una imatge", type=['png', 'jpeg', 'jpg','webp'])
    if img_bytes is not None :
        raw_bytes = np.asarray(bytearray(img_bytes.read()), dtype=np.uint8)
        img_file = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    if img_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model prediction")

def infer_image(img, size=224):
    model.conf = 0.25
    result = model(img, size=size) 
    result.render()
    image = Image.fromarray(result.ims[0])
    return image


global model
st.set_page_config(layout="wide")
st.title("Detecció i Classificació d'Animals")
model = load_model('best.pt', 'cpu')
image_input()


