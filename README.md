Neural Radiosity
---

Code release for the paper, *Neural Radiosity*, accepted to SIGGRAPH Asia 2021 TOG.

[Project Homepage](https://saeedhd96.github.io/neural-radiosity/)

```
Saeed Hadadan, University of Maryland, College Park
Shuhong Chen, University of Maryland, College Park
Matthias Zwicker, University of Maryland, College Park
```

## Environment Setup

Prepare an environment with CUDA 11.7.
Then, in virtualenv or Conda, install PyTorch and other dependencies:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r init/requirements.txt
```

Newer versions of CUDA and PyTorch and Mitsuba should work but have not been tested.

### Installing OpenEXR

1. Linux users should install `libopenexr-dev`
1. Windows users should use Conda and run `conda install -c conda-forge openexr`

## How to Train

Download our data from [Google Drive](https://drive.google.com/drive/folders/1UE4ESxgXK4uL2_f91GqEKmWrvWw2FX8C?usp=sharing).

Training scripts can be found in `./sample_scripts`. Copy them to `./scripts` and edit the data and output paths.

As an exmple, to train for living room scene, run **in this folder**:

```bash
source ./init/init.source  # do this once per shell

bash ./scripts/all_scenes/living_room.sh  # our method
```

We have prepared scripts to train for all scenes, using different encodings:

```bash
source ./init/init.source  # do this once per shell

bash ./scripts/all_scenes/all_sparse_grid.sh  # our method using sparse grids
bash ./scripts/all_scenes/all_hash_grid.sh  # Multi resolution hash encoing by Muller et al. [2022]
bash ./scripts/all_scenes/all_dense_grid.sh  # our method using dense grids

```


## How to Evaluate

Suppose the training folder is `/output/nerad/2023-05-28-22-13-30-living_room`, simply run:

```bash
python test.py \
    test_rendering.image.spp=2048 \
    test_rendering.image.width=512 \
    blocksize=32 \
    experiment=/output/nerad/2023-05-28-22-13-30-living_room
```

All views in the dataset is rendered to `$TRAINING_FOLDER/test/latest`. Check `test.py` and `nerad/model/config.py` for available options.

## Cite

```bibtex
@article{10.1145/3478513.3480569,
author = {Hadadan, Saeed and Chen, Shuhong and Zwicker, Matthias},
title = {Neural Radiosity},
year = {2021},
issue_date = {December 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {40},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3478513.3480569},
doi = {10.1145/3478513.3480569},
abstract = {We introduce Neural Radiosity, an algorithm to solve the rendering equation by minimizing the norm of its residual, similar as in classical radiosity techniques. Traditional basis functions used in radiosity, such as piecewise polynomials or meshless basis functions are typically limited to representing isotropic scattering from diffuse surfaces. Instead, we propose to leverage neural networks to represent the full four-dimensional radiance distribution, directly optimizing network parameters to minimize the norm of the residual. Our approach decouples solving the rendering equation from rendering (perspective) images similar as in traditional radiosity techniques, and allows us to efficiently synthesize arbitrary views of a scene. In addition, we propose a network architecture using geometric learnable features that improves convergence of our solver compared to previous techniques. Our approach leads to an algorithm that is simple to implement, and we demonstrate its effectiveness on a variety of scenes with diffuse and non-diffuse surfaces.},
journal = {ACM Trans. Graph.},
month = {dec},
articleno = {236},
numpages = {11},
keywords = {neural rendering, neural radiance field}
}```
