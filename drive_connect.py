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

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive']

@st.cache(persist=True)
def load_cpu_model():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)

    model_response = service.files().list(q="name = 'model_1016_0.2_new_cpu'", fields="nextPageToken, files(id, name)").execute()
    model_items = model_response.get('files' , [])
    model_id = model_items[0].get('id')
    request = service.files().get_media(fileId=model_id)
    fh = io.BytesIO()
    # fh = open('file', 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
    # fh.close()
    buffer = io.BytesIO(fh.getvalue())
    checkpoint_cpu = torch.load(buffer)

    model_cpu = ImageProcessingTransformer(
        patch_size=4, depth=6, num_heads=4, ffn_ratio=4, qkv_bias=True,drop_rate=0.2, attn_drop_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), )
    model_cpu.load_state_dict(checkpoint_cpu['model_state_dict'])
    print('model_loaded')
    return model_cpu
    

if __name__ == '__main__':
    load_cpu_model()