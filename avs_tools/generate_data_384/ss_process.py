import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ss_root = "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_semantic"

ss_csv_path = "/media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_semantic/metadata.csv"

def resize_img(crop_size, img, is_mask=False):
    outsize = crop_size
    if not is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img


if __name__ == '__main__':
    df_avss = pd.read_csv(ss_csv_path, sep=',')
    # Process ALL subsets (v1m, v1s, v2) and ALL splits (train, val, test)
    print(f"Processing ALL {len(df_avss)} videos (v1m, v1s, v2 x train, val, test)")
    
    for label in ['v1m', 'v1s', 'v2']:
        df_label = df_avss[df_avss['label'] == label]
        print(f"\n=== Processing {label}: {len(df_label)} videos ===")
        
        for index in tqdm(range(len(df_label)), desc=f"{label}"):
            df_one_video = df_label.iloc[index]
            video_name, split, label = df_one_video['uid'], df_one_video['split'], df_one_video['label']

            video_dir = os.path.join(ss_root, label, video_name)

            # image
            T = 10 if label == 'v2' else 5
            img_dir = os.path.join(video_dir, 'frames')
            img_384_dir = os.path.join(video_dir, 'processed_frames_384')
            os.makedirs(img_384_dir, exist_ok=True)
            for i in range(T):
                img_path = os.path.join(img_dir, f'{i}.jpg')
                img_384_path = os.path.join(img_384_dir, f'{i}.jpg')
                
                # Skip if already processed
                if os.path.exists(img_384_path):
                    continue

                try:
                    img = Image.open(img_path)
                    img_384 = resize_img(384, img, is_mask=False)
                    img_384.save(img_384_path)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
            
            # pre mask
            T = 10 if label == 'v2' else 5
            pre_mask_dir = os.path.join(ss_root, 'pre_SAM_mask/AVSBench_semantic', label, video_name, 'frames')
            pre_mask_384_dir = os.path.join(ss_root, 'pre_SAM_mask/AVSBench_semantic', label, video_name, 'processed_frames_384')
            os.makedirs(pre_mask_384_dir, exist_ok=True)
            for i in range(T):
                pre_mask_path = os.path.join(pre_mask_dir, f'{i}_mask_color.png')
                pre_mask_384_path = os.path.join(pre_mask_384_dir, f'{i}_mask_color.png')
                
                # Skip if already processed
                if os.path.exists(pre_mask_384_path):
                    continue

                try:
                    pre_mask = Image.open(pre_mask_path)
                    pre_mask_384 = resize_img(384, pre_mask, is_mask=True)
                    pre_mask_384.save(pre_mask_384_path)
                except Exception as e:
                    print(f"Error processing pre_mask {pre_mask_path}: {e}")
            
            # gt (only for val and test, train has no labels)
            if split in ['val', 'test']:
                T = 10 if label == 'v2' else 5
                gt_dir = os.path.join(video_dir, 'labels_semantic') 
                gt_384_dir = os.path.join(video_dir, 'processed_labels_semantic_384')
                os.makedirs(gt_384_dir, exist_ok=True)
                for i in range(T):
                    gt_path = os.path.join(gt_dir, f'{i}.png')
                    gt_384_path = os.path.join(gt_384_dir, f'{i}.png')
                    
                    # Skip if already processed
                    if os.path.exists(gt_384_path):
                        continue

                    try:
                        gt = Image.open(gt_path)
                        gt_384 = resize_img(384, gt, is_mask=True)
                        gt_384.save(gt_384_path)
                    except Exception as e:
                        print(f"Error processing gt {gt_path}: {e}")

                # rgb gt
                T = 10 if label == 'v2' else 5
                rgb_gt_dir = os.path.join(video_dir, 'labels_rgb') 
                rgb_gt_384_dir = os.path.join(video_dir, 'processed_labels_rgb_384')
                os.makedirs(rgb_gt_384_dir, exist_ok=True)
                for i in range(T):
                    rgb_gt_path = os.path.join(rgb_gt_dir, f'{i}.png')
                    rgb_gt_384_path = os.path.join(rgb_gt_384_dir, f'{i}.png')
                    
                    # Skip if already processed
                    if os.path.exists(rgb_gt_384_path):
                        continue

                    try:
                        rgb_gt = Image.open(rgb_gt_path)
                        rgb_gt_384 = resize_img(384, rgb_gt, is_mask=True)
                        rgb_gt_384.save(rgb_gt_384_path)
                    except Exception as e:
                        print(f"Error processing rgb_gt {rgb_gt_path}: {e}")
    
    print("\n=== Done! ===")
