import os
import sys
import argparse
import torch
from tqdm import tqdm
import numpy as np
import imageio.v2 as imageio
import torch.nn.functional as F

sys.path.append(os.path.join(sys.path[0], '..'))
from DPT.dpt.models import DPTDepthModel

def dpt_depth():
    torch.manual_seed(0)
    is_cuda = (torch.cuda.is_available())
    device = torch.device("cuda" if is_cuda else "cpu")

    path = 'model_zoo/dpt_hybrid-midas-501f0c75.pt'
    non_negative = True
    scale = 0.000305
    shift = 0.1378
    invert = True
    freeze = True
    DPT_model = DPTDepthModel(path, non_negative, scale, shift, invert, freeze).to(device)
    DPT_model.eval()

    
    image_names = [file_name for file_name in os.listdir("data/IMG_6") if file_name.startswith('IMG_')]
    image_paths = [os.path.join("data/IMG_6", file_name) for file_name in image_names]
    
    for i in tqdm(range(len(image_paths)), desc='Processing images'):  
        image_path = image_paths[i] 
        image_name = image_names[i]
        img = torch.from_numpy(imageio.imread(image_path)[...,:3]/255.0).unsqueeze(0).to(torch.float32).to(device).permute(0,3,1,2)
        
        resize_img = F.interpolate(img, size=(1600,2400), mode='bilinear', align_corners=False)
        idx = image_name[4:8]
        depth = DPT_model(resize_img)
        # np.savez(os.path.join(depth_save_dir, 'depth_{}.npz'.format(idx)), pred=depth.detach().cpu())
        depth_array = depth[0].detach().cpu().numpy()
        imageio.imwrite(os.path.join(
            "data/IMG_6", 
            'depth_{}.png'.format(idx)), 
            np.clip(255.0 / depth_array.max() * (depth_array - depth_array.min()), 0, 255).astype(np.uint8))        
 
                                                                
if __name__=='__main__':
    dpt_depth()