# SKA-VCT: Listening to the Motion Probing Physical Consistency for Audio-Visual Segmentation

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-blue)](#)
[![Code Status](https://img.shields.io/badge/Code-Under%20Preparation-orange)](#)

This is the official repository for the paper **"Listening to the Motion: Probing Physical Consistency for Audio-Visual Segmentation"**.

> **Note**: The code is currently under organization and cleaning. The complete source code, including the model definition, training scripts, and pretrained weights, will be released here soon. Please stay tuned!

## Introduction

Audio-Visual Segmentation (AVS) requires precise pixel-level localization of sounding objects. While the emerging vision-centric paradigm has improved boundary delineation, it fundamentally suffers from a dependency on static visual saliency. This often leads to false positives where salient but silent objects are segmented due to a lack of dynamic verification. Conversely, naive attempts to incorporate motion cues via optical flow often introduce uncorrelated noise from background dynamics. 

In response, we propose **SKA-VCT**, a robust framework bridging this gap by enforcing physical consistency. At the core lies the **Spectral-Kinematic Alignment (SKA)** module, which utilizes audio spectral features as queries to selectively retrieve motion representations from a pre-computed kinematic field, yielding a sound-activated motion map that inherently suppresses background dynamics. This motion prior is subsequently leveraged by a **Motion-Prompted Query Generation (MPQG)** module to dynamically reweight visual features, guiding the object queries to focus on genuine sounding objects. Finally, to mitigate the boundary ambiguity caused by low-resolution kinematic features, an auxiliary **Boundary Refinement Module (BRM)** is introduced to refine object contours. 

Extensive experiments on the AVSBench dataset demonstrate the superiority of SKA-VCT across various settings.

## TODOs
- [ ] Initial Code Commit
- [ ] Pre-trained Models Release
- [ ] Data Preparation Scripts
- [ ] Training and Evaluation Instructions

## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@article{ska_vct_2025,
  title={Listening to the Motion: Probing Physical Consistency for Audio-Visual Segmentation},
  author={Anonymous},
  year={2025}
}
```

## Acknowledgement
This codebase is built upon [VCT_AVS](https://github.com/spyflying/VCT_AVS) and [COMBO-AVS](https://github.com/yannqi/COMBO-AVS). We thank the authors for their great work.