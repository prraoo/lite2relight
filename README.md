<p align="center">

  <h1 align="center">Lite2Relight: 3D-aware Single Image Portrait Relighting
    <a href='https://doi.org/10.1145/3641519.3657470'>
    <img src='https://img.shields.io/badge/Paper-(58 MB)-red' alt='PDF'>
    </a>
    <a href='https://vcai.mpi-inf.mpg.de/projects/Lite2Relight/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    <a href='https://arxiv.org/abs/2407.10487'>
    <img src='https://img.shields.io/badge/Arxiv-red' alt='arxiv PDF'>
    </a>
  </h1>
  <h2 align="center">ACM SIGGRAPH 2024 Conference Proceedings</h2>
  <div align="center">
  </div>
</p>
<p float="center">
  <img src="assets/teaser.png" width="98%" />
</p>

This repository contains the official implementation for Lite2Relight.

## Requirements

* We recommend Linux for performance and compatibility reasons.
* 1 high-end NVIDIA GPU.
* 64-bit Python 3.10 and PyTorch 2.5.1 (or later).
* CUDA toolkit 12.1 or later.
* Python libraries: see [requirements.txt](./requirements.txt) for library dependencies. You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda create --name lite2relight --file requirements.txt`
  - `conda activate lite2relight`

## Getting started

Pre-trained networks are stored as `*.pkl` or `*.pt` files that can be referenced using local filenames. Please download the models from [this Google Drive link](https://drive.google.com/file/d/1WnoLv_2O4sXaBe-FEKEtvRxzBZn5IpVE/view?usp=sharing). After downloading `pretrained.zip`, extract it and place the `checkpoints` and `pretrained_models` folders in the main directory of this repository.

## Sample Dataset

The `sample` directory provides a quick way to get started. It is structured as follows:

```
sample/
├── dataset
│   └── ID00600
├── envmaps
│   ├── EMAP-0059.png
│   ├── ...
│   └── EMAP-335.png
└── in-the-wild
    └── ID00600.jpg
```

### Data Preprocessing

1.  **In-the-wild images**: The `sample/in-the-wild` folder contains the test images.
2.  **Preprocessing**: To process these images, please follow the data processing steps for in-the-wild portraits as described in the [EG3D repository](https://github.com/NVlabs/eg3d/blob/main/README.md#preparing-datasets). After preprocessing, you should have a processed folder like `sample/dataset/ID00600`.
3.  **Environment Maps**: The `sample/envmaps` folder contains sample environment maps. These are 20x10 HDR environment maps in `.png` format. You can downsample your own HDR environment maps to this format.

## Inference

You can run inference on the sample data using the provided script. Before running, make sure to update the checkpoint paths in `scripts/inference.sh`.

```bash
bash scripts/inference.sh
```

This will run `infer_relit.py` with the appropriate arguments and save the results in the `results/release/` directory.

## Training

Training code will be released using the FaceOLAT dataset, which can be found at the [3DPR project page](https://vcai.mpi-inf.mpg.de/projects/3dpr/).

## Acknowledgements

This work is built upon the following amazing projects. We thank the authors for their great work.

*   [EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks](https://github.com/NVlabs/eg3d/tree/main)
*   [GOAE: A Triptych of GOAEs for Cross-Domain Man-in-the-Middle Attacks and Defenses](https://github.com/jiangyzy/GOAE)


## Citation
If you find our code or paper useful, please cite as:
```
@article{prao2024lite2relight,
title = {Lite2Relight: 3D-aware Single Image Portrait Relighting},
author = {Rao, Pramod and Fox, Gereon and Meka, Abhimitra and B R, Mallikarjun and Zhan, Fangneng and Weyrich, Tim and Bickel, Bernd and Seidel, Hans-Peter and Pfister, Hanspeter and Matusik, Wojciech and Elgharib, Mohamed and Theobalt, Christian },
booktitle = {ACM SIGGRAPH 2024 Conference Proceedings},
year={2024}
}
```
