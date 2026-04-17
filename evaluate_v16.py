import cv2
import numpy as np
import os
import glob

def get_variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def check_blur(run_name, video_idx, kf_idx):
    if run_name == 'baseline':
        path = f"./output/Wild_SLAM_iPhone/iphone_wandering/plots_after_refine/video_idx_{video_idx}_kf_idx_{kf_idx}.png"
    elif run_name == 'temporal':
        path = f"./output/Wild_SLAM_iPhone_temporal/iphone_wandering_temporal/plots_after_refine/video_idx_{video_idx}_kf_idx_{kf_idx}.png"
    else:
        path = f"./output/Wild_SLAM_iPhone_temporal_objperm_{run_name}/iphone_wandering_temporal_objperm_{run_name}/plots_after_refine/video_idx_{video_idx}_kf_idx_{kf_idx}.png"
    
    if not os.path.exists(path):
        return None
        
    img = cv2.imread(path)
    if img is None:
        return None
        
    H, W = img.shape[:2]
    # panel is bottom half, right half is render
    render = img[H//2:, W//2:]
    
    # Let's crop the far background (e.g. wall at the end of the corridor)
    # Using approximate coordinates from earlier
    hr, wr = render.shape[:2]
    far_bg = render[int(hr*0.3):int(hr*0.5), int(wr*0.4):int(wr*0.6)]
    
    gray = cv2.cvtColor(far_bg, cv2.COLOR_BGR2GRAY)
    return get_variance_of_laplacian(gray)

runs = ['baseline', 'temporal', 'v7_smoke', 'v15_smoke', 'v16_smoke']
frames = [(10, 90), (14, 126), (19, 171)]

for f in frames:
    print(f"Frame {f[0]}_{f[1]}:")
    for r in runs:
        v = check_blur(r, f[0], f[1])
        print(f"  {r:10s}: {v:.2f}" if v is not None else f"  {r:10s}: N/A")
