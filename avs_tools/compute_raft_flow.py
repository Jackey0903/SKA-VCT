"""
[Ours: RAFT Flow] 使用 torchvision 内置 RAFT 预计算光流

Usage:
    # S4 数据集
    python avs_tools/compute_raft_flow.py \
        --dataset s4 \
        --split train \
        --input_root /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Single-source \
        --output_root /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/raft_flow
    
    # MS3 数据集
    python avs_tools/compute_raft_flow.py \
        --dataset ms3 \
        --split train \
        --input_root /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/AVSBench_object/Multi-sources \
        --output_root /media/a100/c5e1bf65-7974-432f-8aed-7a1345241efe/hhj/AVS_dataset/raft_flow

Author: [Ours]
Date: 2024-11-28
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms import functional as F


def load_model(device):
    """加载 torchvision 内置的 RAFT 模型"""
    print("Loading RAFT model from torchvision...")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True)
    model = model.to(device).eval()
    print("RAFT model loaded!")
    return model, weights.transforms()


def load_image_pair(path1, path2, transforms, device):
    """加载并预处理图像对"""
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    
    # 转换为 tensor
    img1_tensor = F.to_tensor(img1).unsqueeze(0).to(device)  # [1, 3, H, W]
    img2_tensor = F.to_tensor(img2).unsqueeze(0).to(device)
    
    # 应用 RAFT 预处理 (归一化等)
    img1_batch, img2_batch = transforms(img1_tensor, img2_tensor)
    
    return img1_batch, img2_batch, img1.size  # (W, H)


def compute_flow_magnitude(flow):
    """
    将光流转换为 magnitude (单通道)
    
    Args:
        flow: [2, H, W] tensor, (u, v) 光流场
    
    Returns:
        magnitude: [H, W] numpy array, 归一化到 [0, 1]
    """
    u = flow[0].cpu().numpy()
    v = flow[1].cpu().numpy()
    magnitude = np.sqrt(u**2 + v**2)
    
    # 归一化到 [0, 1]
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = magnitude / max_val
    
    return magnitude.astype(np.float32)


def process_video(model, transforms, video_path, output_path, device):
    """处理单个视频的所有帧"""
    os.makedirs(output_path, exist_ok=True)
    
    # 获取所有帧
    frames = sorted([f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(frames) == 0:
        print(f"Warning: No frames found in {video_path}")
        return
    
    for i in range(len(frames)):
        output_file = os.path.join(output_path, f'flow_{i:04d}.npy')
        
        # 跳过已计算的
        if os.path.exists(output_file):
            continue
        
        if i == 0:
            # 第一帧：使用零光流
            img = Image.open(os.path.join(video_path, frames[0]))
            w, h = img.size
            flow_mag = np.zeros((h, w), dtype=np.float32)
        else:
            # 计算 frame[i-1] -> frame[i] 的光流
            path1 = os.path.join(video_path, frames[i-1])
            path2 = os.path.join(video_path, frames[i])
            
            img1, img2, orig_size = load_image_pair(path1, path2, transforms, device)
            
            with torch.no_grad():
                # RAFT 返回 list of flow predictions (迭代 refinement)
                # 取最后一个 (最精确)
                flow_predictions = model(img1, img2)
                flow = flow_predictions[-1][0]  # [2, H, W]
            
            flow_mag = compute_flow_magnitude(flow)
            
            # 如果尺寸不匹配，resize 回原始尺寸
            if flow_mag.shape != (orig_size[1], orig_size[0]):
                from PIL import Image as PILImage
                flow_pil = PILImage.fromarray(flow_mag)
                flow_pil = flow_pil.resize(orig_size, PILImage.BILINEAR)
                flow_mag = np.array(flow_pil)
        
        np.save(output_file, flow_mag)


def load_ms3_split_info(input_root):
    """加载 MS3 meta_data.csv 获取 split 信息"""
    import csv
    meta_path = os.path.join(input_root, 'ms3_meta_data.csv')
    split_dict = {'train': [], 'val': [], 'test': []}
    
    with open(meta_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过 header
        for row in reader:
            video_id, split = row[0], row[1]
            if split in split_dict:
                split_dict[split].append(video_id)
    
    return split_dict


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model, transforms = load_model(device)
    
    # 数据集路径配置
    if args.dataset == 's4':
        frame_dir = 's4_data_384/visual_frames_384'
        # S4 结构: visual_frames_384/split/category/video/frame.png
        has_category = True
        ms3_split_dict = None
    elif args.dataset == 'ms3':
        frame_dir = 'ms3_data_384/visual_frames_384'
        # MS3 结构: visual_frames_384/video/frame.png (所有视频在一个目录)
        has_category = False
        # 加载 MS3 split 信息
        ms3_split_dict = load_ms3_split_info(args.input_root)
        print(f"MS3 split info loaded: train={len(ms3_split_dict['train'])}, val={len(ms3_split_dict['val'])}, test={len(ms3_split_dict['test'])}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # 处理指定 split
    splits = [args.split] if args.split != 'all' else ['train', 'val', 'test']
    
    for split in splits:
        if has_category:
            # S4: split 是子目录
            input_path = os.path.join(args.input_root, frame_dir, split)
        else:
            # MS3: 所有视频在同一目录
            input_path = os.path.join(args.input_root, frame_dir)
        
        output_base = os.path.join(args.output_root, args.dataset, split)
        
        if not os.path.exists(input_path):
            print(f"Warning: Input path does not exist: {input_path}")
            continue
        
        if has_category:
            # S4: 遍历 category/video
            categories = sorted(os.listdir(input_path))
            total_videos = 0
            for cat in categories:
                cat_path = os.path.join(input_path, cat)
                if os.path.isdir(cat_path):
                    total_videos += len(os.listdir(cat_path))
            
            print(f"\nProcessing {args.dataset}/{split}: {len(categories)} categories, {total_videos} videos")
            
            for category in tqdm(categories, desc=f'{args.dataset}/{split}'):
                category_input = os.path.join(input_path, category)
                if not os.path.isdir(category_input):
                    continue
                
                videos = sorted(os.listdir(category_input))
                for video in videos:
                    video_input = os.path.join(category_input, video)
                    video_output = os.path.join(output_base, category, video)
                    
                    if not os.path.isdir(video_input):
                        continue
                    
                    process_video(model, transforms, video_input, video_output, device)
        else:
            # MS3: 通过 meta_data 过滤指定 split 的视频
            videos_in_split = ms3_split_dict[split]
            print(f"\nProcessing {args.dataset}/{split}: {len(videos_in_split)} videos")
            
            for video in tqdm(videos_in_split, desc=f'{args.dataset}/{split}'):
                video_input = os.path.join(input_path, video)
                video_output = os.path.join(output_base, video)
                
                if not os.path.isdir(video_input):
                    print(f"Warning: Video directory not found: {video_input}")
                    continue
                
                process_video(model, transforms, video_input, video_output, device)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute RAFT optical flow for AVS datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['s4', 'ms3', 'avss'],
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to process')
    parser.add_argument('--input_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Root directory to save optical flow')
    
    args = parser.parse_args()
    main(args)

