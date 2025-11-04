# Interpretable Generative Models through Post-hoc Concept Bottlenecks (CVPR 2025)

### [Paper](https://arxiv.org/abs/2503.19377) | [Project Page](https://lilywenglab.github.io/posthoc-generative-cbm/)

This is the official repository for the **CVPR 2025** paper: [Interpretable Generative Models through Post-hoc Concept Bottlenecks](https://arxiv.org/abs/2503.19377)

* We propose two novel methods to enable interpretability for generative models:
    * **CB-AE**: The 1st method for post-hoc interpretable generative models. **CB-AE** can be trained efficiently with a frozen pretrained generative model, without real concept-labeled images.
    * **Concept Controller**: An optimization-based concept intervention method with improved steerability and higher image quality.
* We show our methods have **higher steerability** (+31% and +28% better than prior SOTA) and **lower cost** (4-15x faster to train) on deep generative models including GANs and diffusion models. 

<p align="center">
    <img src="https://lilywenglab.github.io/posthoc-generative-cbm/assets/fig1_teaser_website_example.svg" width="90%" alt="Overview">
</p>

## Table of Contents

* [Setup](#setup)
    * [Environment setup instructions](#environment-setup-instructions)
    * [Download base model and CB-AE/CC weights](#download-base-model-and-cb-aecc-weights)
    * [Download concept classifier weights](#download-concept-classifier-weights)
* [Demo](#demo)
* [Training](#training)
* [Evaluation](#evaluation)
* [Results](#results)
* [Sources](#sources)
* [Cite this work](#cite-this-work)

## Setup

### Environment setup instructions

* Conda environment installation:
    ```
    conda create -n posthocgencbm python=3.8
    conda install nvidia/label/cuda-11.7.0::cuda-nvcc cudatoolkit
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install -r requirements.txt
    ```

* Download the CelebA-HQ-pretrained StyleGAN2 base model from [below](#download-base-model-and-cb-aecc-weights) and test the environment using `python3 eval/test_stygan2.py`. It should save a StyleGAN2 generated image in `images/`. If you get CUDA runtime errors (during "Setting up PyTorch plugin..."), use this:
    ```
    export CUDA_HOME=$CONDA_PREFIX
    export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
    export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
    ```

### Download base model and CB-AE/CC weights

* We use `models/checkpoints` for saving/loading CB-AE/CC checkpoints
    ```
    mkdir models/checkpoints
    cd models/checkpoints
    ```

* CelebA-HQ-pretrained StyleGAN2 (from [[2]](#sources)):
    ```
    ## base model weights
    wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-celebahq-256x256.pkl
    ## CB-AE weights
    gdown https://drive.google.com/uc?id=1RBdjcBDbpAoW5qOkG-rBonIpcBApBF-q
    ## CC weights
    gdown https://drive.google.com/uc?id=1fh2XV2ttrCc88-SgfR9f-JcwG1eent_U
    ```

* CUB-pretrained StyleGAN2 (trained using [[4]](#sources)):
    ```
    ## base model weights
    gdown https://drive.google.com/uc?id=1sW7WgvUFH2REZPQx88BjFneoItP9C0XB
    ```

* CelebA-HQ-pretrained DDPM-256x256 (from [[3]](#sources)):
    ```
    ## base model weights get downloaded automatically via HuggingFace when using "-e cbae_ddpm" (i.e. any config using this DDPM model)
    ## CB-AE weights
    gdown https://drive.google.com/uc?id=1kl5pDqzm0M73r8H74AfSokgDFGAF0szb
    ```

### Download concept classifier weights

* [ResNet18 CelebA-HQ](https://drive.google.com/uc?id=1xbR7MbERV7wMnU4WcsNSDriYXBqsy_jZ)-based classifiers for training and visualization.
* [ViT-L-16 CelebA-HQ](https://drive.google.com/uc?id=1XD6Badmf4QwRrdy6MbOr-mefyu1k_OIy)-based classifiers for quantitative evaluation.
    ```
    cd models/checkpoints
    ## ResNet18 CelebA-HQ
    gdown https://drive.google.com/uc?id=1xbR7MbERV7wMnU4WcsNSDriYXBqsy_jZ
    unzip celebahq_rn18_conclsf.zip
    ## ViT-L-16 CelebA-HQ (relatively large file of ~8.4 GB, so download only if you want to do evaluations)
    gdown https://drive.google.com/uc?id=1XD6Badmf4QwRrdy6MbOr-mefyu1k_OIy
    unzip celebahq_vitl16_conclsf.zip
    ## ResNet18 CelebA (64x64)
    gdown https://drive.google.com/uc?id=15m6xCI5JPZaz-BaSoCjHCJCeof53G4rC
    unzip celeba64_rn18_conclsf.zip
    ## ResNet50 CUB (256x256)
    gdown https://drive.google.com/uc?id=1vW5Q41FGHXdTqbraz54AXQ2uoBKispLD
    unzip cub_rn50_conclsf.zip
    ## ResNet50 CUB (64x64)
    gdown https://drive.google.com/uc?id=1vvlWd4MWB62-lyq2sPQAVhf5Pqc5Mnzf
    unzip cub64_rn50_conclsf.zip
    ```
* Other concept classifiers can be trained using `train/train_conclsf.py`.

## Demo

* Follow `notebooks/visualize_interventions.ipynb` for concept interventions demo with CelebA-HQ-pretrained StyleGAN2 with CB-AE.
    * Note: Before the notebook, please [download](#download-concept-classifier-weights) concept classifier weights (ResNet18) for visualization.


## Training

* Use `bash scripts/train_cbae.sh` to train a CB-AE for a CelebA-HQ-pretrained StyleGAN2 with supervised classifiers as pseudo-label source.
* Some important arguments to specify are:
    * `-e`: specify which config file from the `config/` folder to use (e.g. `cbae_stygan2`).
    * `-d`: specify dataset of base generative model (e.g. `celebahq`).
    * `-t`: specify experiment name to be used as a suffix for saving logs, checkpoints, etc.
    * `-p`: specify pseudo-label source $M$ for CB-AE/CC training (e.g. `supervised` for supervised-trained classifiers, `clipzs` for zero-shot CLIP classifiers, `tipzs` for few-shot adapted CLIP).
* The same `train_cbae.sh` has commented out examples for training Concept Controller (CC).

## Evaluation

* Use `bash scripts/eval_intervention.sh` for an example that runs steerability evaluation for `Smiling` concept for a CelebA-HQ StyleGAN2 CB-AE.
* Some important arguments to specify are:
    * `-e`, `-d`, and `-t` should be the same as from training (or use based on downloaded CB-AE/CC checkpoint, e.g. `celebahq_cbae_stygan2_thr90_sup_pl_cls8_cbae.pt` would use `-d celebahq -e cbae_stygan2_thr90 -t sup_pl_cls8`).
    * `-c`: concept to intervene on (e.g. `Smiling` or `Mouth_Slightly_Open`).
    * `-v`: desired concept value (e.g. `0` or `1` based on if desired target concept is `Smiling` or `Not Smiling`).
    * `--optint`: use this for optimization-based interventions (not using this will use CB-AE interventions).
    * `--visualize`: use this to visualize some examples (not using this will run the full quantitative evaluation).
* The same `eval_intervention.sh` has commented out examples for evaluating StyleGAN2 CC and DDPM CB-AE.

## Results

### 1. Concept Steerability or Intervention Success Rate
* Our CB-AE and CC improves steerability across GANs (+31%) and diffusion models (+28%) over the prior state-of-the-art method CBGM [[1]](#sources) while being 4-15x faster to train on average.
<p align="center">
    <img src="https://lilywenglab.github.io/posthoc-generative-cbm/assets/table3_posthocgencbm.png" width="90%" alt="Steerability Evaluation">
</p>

### 2. Interpreting generated images
* Our CB-AE (and CC) provide human-understandable concept predictions along with the generated images.
<p align="center">
    <img src="https://lilywenglab.github.io/posthoc-generative-cbm/assets/fig5_cbae_conc_examples.svg" width="90%" alt="Concept Predictions">
</p>

### 3. Concept intervention examples (CB-AE interventions)
* Concept intervention (modifying concepts) in the CB-AE leads to appropriate changes in the resulting image generation, enabling controllable generation. 
<p align="center">
    <img src="https://lilywenglab.github.io/posthoc-generative-cbm/assets/fig6_cbae_interv_examples.png" width="90%" alt="CB-AE Interventions">
</p>

### 4. Concept intervention examples (optimization-based interventions)
* Optimization-based interventions also enable controllable generation with improved orthogonality (*i.e.* less change in other concepts, closer to the original generation) than CB-AE interventions.
<p align="center">
    <img src="https://lilywenglab.github.io/posthoc-generative-cbm/assets/fig7_optint_examples.png" width="90%" alt="Optimization-based Interventions">
</p>

## Sources

[1] CBGM (ICLR 2024): [https://github.com/prescient-design/CBGM](https://github.com/prescient-design/CBGM)

[2] [StyleGAN3 GitHub repo](https://github.com/NVlabs/stylegan3?tab=readme-ov-file#additional-material) (it has StyleGAN2 pretrained weights for CelebA-HQ and CUB)

[3] [CelebA-HQ pretrained DDPM repo](https://huggingface.co/google/ddpm-celebahq-256)

[4] [StyleGAN2-Ada PyTorch GitHub repo](https://github.com/NVlabs/stylegan2-ada-pytorch)

## Cite this work

A. Kulkarni, G. Yan, C. Sun, T. Oikarinen, and T.-W. Weng, [Interpretable Generative Models through Post-hoc Concept Bottlenecks](https://arxiv.org/abs/2503.19377), CVPR 2025

```
@inproceedings{kulkarni2025interpretable
    title={Interpretable Generative Models through Post-hoc Concept Bottlenecks},
    author={Kulkarni, Akshay and Yan, Ge and Sun, Chung-En and Oikarinen, Tuomas and Weng, Tsui-Wei},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025},
}
```
