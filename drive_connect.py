from __future__ import print_function
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload
import sys
import io
import torch
from ipt import ImageProcessingTransformer
from functools import partial
import torch.nn as nn
import streamlit as st
import requests as req
from urllib import request

@st.cache(persist=True)
def load_cpu_model():
    URL = 'https://storage.cloud.google.com/terraform-test-336308-model-bucket/model_1016_0.2_new_cpu?authuser=3'
    res = request.urlopen(URL).read()
    file = io.BytesIO(res)
    checkpoint_cpu = torch.load(file)

    model_cpu = ImageProcessingTransformer(
        patch_size=4, depth=6, num_heads=4, ffn_ratio=4, qkv_bias=True,drop_rate=0.2, attn_drop_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), )
    model_cpu.load_state_dict(checkpoint_cpu['model_state_dict'])
    print('model_loaded')
    return model_cpu
    

if __name__ == '__main__':
    load_cpu_model()