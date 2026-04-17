import os
import glob
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

def get_deeplab_model():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def evaluate_person_pixels(run_name, model):
    if run_name == 'baseline':
        pattern = "./output/Wild_SLAM_iPhone/iphone_wandering/plots_after_refine/video_idx_*_kf_idx_*.png"
    elif run_name == 'temporal':
        pattern = "./output/Wild_SLAM_iPhone_temporal/iphone_wandering_temporal/plots_after_refine/video_idx_*_kf_idx_*.png"
    else:
        pattern = f"./output/Wild_SLAM_iPhone_temporal_objperm_{run_name}/iphone_wandering_temporal_objperm_{run_name}/plots_after_refine/video_idx_*_kf_idx_*.png"
        
    files = sorted(glob.glob(pattern))
    if not files:
        return None
        
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    total_pixels = 0
    count = 0
    
    # Process only a subset to be fast (e.g. keyframes with people, typically frame 15-30)
    for path in files:
        fname = os.path.basename(path)
        vidx = int(fname.split('_')[2])
        if vidx < 15 or vidx > 30:
            continue
            
        img = cv2.imread(path)
        H, W = img.shape[:2]
        render = img[H//2:, W//2:]
        render_rgb = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
        
        input_tensor = preprocess(render_rgb)
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))['out'][0]
            output_predictions = output.argmax(0).byte().cpu().numpy()
            
        person_mask = (output_predictions == 15).astype(np.uint8)
        total_pixels += person_mask.sum()
        count += 1
        
    return total_pixels / max(count, 1)

print("Loading model...")
model = get_deeplab_model()
runs = ['baseline', 'v15_smoke', 'v16_smoke']

for r in runs:
    avg_pixels = evaluate_person_pixels(r, model)
    print(f"{r:15s}: {avg_pixels:.1f} avg person pixels (frames 15-30)")
