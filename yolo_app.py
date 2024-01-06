import streamlit as st
import torch
from PIL import Image

@st.cache_resource
def load_model(path, device):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model_ = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
    return model_
 
def image_input():
    img_file = None
    img_bytes = st.file_uploader("Tria o arrosega una imatge", type=['png', 'jpeg', 'jpg','webp'])
    if img_bytes is not None :
        img_file = Image.open(img_bytes)
    if img_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file)
        with col2:
            imatge,taula = infer_image(img_file)
            st.image(imatge)
            config = {'xmin':st.column_config.NumberColumn('X1',format='%d',width='small'),
                    'ymin':st.column_config.NumberColumn('Y1',format='%d',width='small'),
                    'xmax':st.column_config.NumberColumn('X2',format='%d',width='small'),
                    'ymax':st.column_config.NumberColumn('Y2',format='%d',width='small'),
                    'confidence':st.column_config.NumberColumn('Confiança',format='%.3f',width='small'),
                    'class':st.column_config.NumberColumn('Classe',format='%d',width='small'),
                    'name':st.column_config.Column('Nom',width='medium')}
            st.dataframe(taula,hide_index=True,column_config=config,width=imatge.size[0])
    
def infer_image(img, size=416):
    model.conf = 0.25
    result = model(img, size=size) 
    result.render()
    image = Image.fromarray(result.ims[0])
    return image,result.pandas().xyxy[0]

global model
st.set_page_config(layout="wide")
st.title("Detecció i Classificació d'Animals")
model = load_model('best.pt', 'cpu')
image_input()


