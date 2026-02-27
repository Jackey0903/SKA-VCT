"""将 SemanticSAM .npy 文件转换为 PNG 格式，用于 ms3_process.py"""
import json
import os 
from detectron2.utils.file_io import PathManager
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import argparse

def ade_palette():
    """ADE20K palette for external use."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_root", type=str, 
                       default='AVS_dataset/pre_SemanticSAM_mask/AVSBench_object/Multi-sources/ms3_data/visual_frames',
                       help="SemanticSAM .npy 文件根目录")
    parser.add_argument("--output_root", type=str,
                       default='AVS_dataset/AVSBench_semantic/pre_SAM_mask/AVSBench_semantic/v1m',
                       help="输出 PNG 文件根目录")
    parser.add_argument("--csv_path", type=str,
                       default='AVS_dataset/AVSBench_semantic/metadata.csv',
                       help="metadata.csv 路径")
    args = parser.parse_args()
    
    # 读取 v1m 视频列表
    df_all = pd.read_csv(args.csv_path, sep=',')
    df_v1m = df_all[df_all['label'] == 'v1m']
    
    print(f"总共需要处理 {len(df_v1m)} 个 v1m 视频")
    
    color_map = np.array(ade_palette()).astype(np.uint8)
    processed = 0
    skipped = 0
    
    for index in tqdm(range(len(df_v1m))):
        df_one_video = df_v1m.iloc[index]
        ori_name = df_one_video['vid']  # 原始视频名称，如 0bzkGQLy7b4_2
        video_name = df_one_video['uid']  # 在 v1m 目录下的名称
        
        # 输入：SemanticSAM .npy 文件目录
        npy_dir = os.path.join(args.npy_root, ori_name)
        
        # 输出：PNG 文件目录
        output_dir = os.path.join(args.output_root, video_name, 'frames')
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查 npy 目录是否存在
        if not os.path.isdir(npy_dir):
            print(f"⚠️ 跳过（无 npy 数据）: {ori_name}")
            skipped += 1
            continue
        
        # 处理每一帧（0-4）
        for i in range(5):
            # SemanticSAM 生成的文件名格式
            npy_candidates = [
                os.path.join(npy_dir, f'{ori_name}.mp4_{i+1}_mask.npy'),
                os.path.join(npy_dir, f'{ori_name}_{i+1}_mask.npy'),
                os.path.join(npy_dir, f'{i}_mask.npy'),
            ]
            
            npy_path = None
            for cand in npy_candidates:
                if os.path.exists(cand):
                    npy_path = cand
                    break
            
            if npy_path is None:
                print(f"⚠️ 缺少文件: {ori_name} frame {i}")
                continue
            
            # 输出 PNG 文件路径
            png_path = os.path.join(output_dir, f'{i}_mask_color.png')
            
            # 如果已存在则跳过
            if os.path.exists(png_path):
                continue
            
            # 加载 .npy 文件
            try:
                pre_mask = np.load(npy_path, allow_pickle=True)
            except Exception as e:
                print(f"❌ 加载失败: {npy_path}, 错误: {e}")
                continue
            
            # 处理 mask 数据
            try:
                pre_mask[0].dtype
            except:
                # 如果 mask 为空，创建空白 mask
                pre_int_mask = np.zeros((1, 224, 224))
            else:
                pre_int_mask = pre_mask.astype(np.uint8)
            
            # 按照面积排序并合并 mask
            sums = np.sum(pre_int_mask, axis=(1, 2))
            sorted_indices = np.argsort(sums)
            sorted_pre_mask = pre_int_mask[sorted_indices]
            
            m = None
            for idx in range(sorted_pre_mask.shape[0]):
                if idx == 0:
                    m = sorted_pre_mask[idx]
                else:
                    m = np.where(m == 0, sorted_pre_mask[idx] * (idx + 1), m)
            
            if m is None:
                m = np.zeros((224, 224), dtype=np.uint8)
            else:
                m = m.astype(np.uint8)
            
            # 保存为 PNG（不调整大小，保持原始分辨率）
            mask_img = Image.fromarray(m)
            mask_img.putpalette(color_map)
            mask_img.save(png_path)
        
        processed += 1
    
    print(f"\n✅ 处理完成！")
    print(f"   成功处理: {processed} 个视频")
    print(f"   跳过（无数据）: {skipped} 个视频")

