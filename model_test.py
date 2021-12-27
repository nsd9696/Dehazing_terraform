from PIL import Image
import numpy as np
from ipt import ImageProcessingTransformer
from functools import partial
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

img = Image.open('hazy.jpg')
model = ImageProcessingTransformer(
    patch_size=4, depth=6, num_heads=4, ffn_ratio=4, qkv_bias=True,drop_rate=0.2, attn_drop_rate=0.2,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), )
checkpoint_cpu = torch.load('model_1016_0.2_cpu')
model.load_state_dict(checkpoint_cpu['model_state_dict'])

def img_transform(image):
    
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((128,128)),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])
    return torch.unsqueeze(trans(image),0)

def inference(model,image):
    model.set_task(5)
    with torch.no_grad():
        model.eval()
        pred_result = torch.squeeze(model(image))
    plt.imshow(np.transpose(pred_result.detach().numpy() * 0.5 + 0.5, (1,2,0)))
    plt.show()
    clip_img = np.clip(np.transpose(pred_result.detach().numpy().astype(float) * 0.5 + 0.5, (1,2,0)),0,1) * 255
    pil_img = clip_img.astype('uint8')
    return Image.fromarray(pil_img)

img_ = img_transform(img)
result = inference(model,img_)

