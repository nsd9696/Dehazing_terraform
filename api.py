import os
import sys
import io
import base64
import torchvision.transforms as transforms
import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt

@st.cache()
def img_transform(image):
    
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((128,128)),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])
    return trans(image).unsqueeze(0)

@st.cache()
def inference(model,image):
    model.set_task(5)
    with torch.no_grad():
        model.eval()
        pred_result = torch.squeeze(model(image))
    clip_img = np.clip(np.transpose(pred_result.detach().numpy().astype(float) * 0.5 + 0.5, (1,2,0)),0,1) * 255
    pil_img = clip_img.astype('uint8')
    return Image.fromarray(pil_img)

@st.cache()
def get_image_download_link(img,filename,text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

    