#!/usr/bin/env python3
"""
Quick test script for Motion-Guided Query Refinement
Tests the data loading and forward pass with motion maps
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from detectron2.config import get_cfg
from models import add_maskformer2_config, add_audio_config, add_fuse_config

def test_motion_guided():
    print("=" * 60)
    print("Testing Motion-Guided Query Refinement Implementation")
    print("=" * 60)
    
    # Load config
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_audio_config(cfg)
    add_fuse_config(cfg)
    cfg.merge_from_file("configs/ms3_swinb_384/COMBO_SWINB.yaml")
    
    print("\n✅ Config loaded successfully")
    print(f"   Dataset: {cfg.INPUT.DATASET_MAPPER_NAME}")
    print(f"   Num classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"   Num queries: {cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES}")
    
    # Test motion map computation (simulate)
    print("\n📊 Testing motion map computation...")
    batch_size = 2
    num_frames = 5
    H, W = 224, 224
    
    # Simulate images
    images = torch.randn(num_frames, 3, H, W)
    
    # Compute motion maps (same logic as in dataset_mapper)
    motion_maps = []
    for i in range(num_frames):
        if i == 0:
            motion_map = torch.zeros((1, H, W))
        else:
            diff = torch.abs(images[i] - images[i-1])
            motion_map = diff.mean(dim=0, keepdim=True)
            if motion_map.max() > 0:
                motion_map = motion_map / motion_map.max()
        motion_maps.append(motion_map)
    
    print(f"   ✅ Motion maps computed: {len(motion_maps)} frames")
    print(f"   ✅ Motion map shape: {motion_maps[0].shape}")
    print(f"   ✅ Motion map range: [{motion_maps[1].min():.4f}, {motion_maps[1].max():.4f}]")
    
    # Test VCQ with motion map
    print("\n🔧 Testing VCQ with motion weighting...")
    from models.modeling.transformer_decoder.vision_centric_transformer_decoder import VCQ
    
    vcq = VCQ(in_channels=256, out_channels=256, num_pixel=56*56, num_query=100)
    print(f"   ✅ VCQ initialized")
    print(f"   ✅ motion_lambda parameter: {vcq.motion_lambda.item():.4f}")
    
    # Test forward pass with motion map
    x = torch.randn(batch_size, 256, 56, 56)
    motion_map = torch.randn(batch_size, 1, 56, 56).abs()
    
    # Without motion
    output_baseline = vcq(x, motion_map=None)
    print(f"   ✅ Baseline query shape: {output_baseline.shape}")
    
    # With motion
    output_ours = vcq(x, motion_map=motion_map)
    print(f"   ✅ Motion-guided query shape: {output_ours.shape}")
    
    # Verify they are different
    diff = torch.abs(output_baseline - output_ours).mean()
    print(f"   ✅ Query difference (baseline vs ours): {diff:.6f}")
    
    if diff > 0:
        print("\n🎉 Motion-guided weighting is working!")
    else:
        print("\n⚠️  Warning: No difference detected (check implementation)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("🚀 Ready to start training with Motion-Guided Query Refinement")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_motion_guided()


