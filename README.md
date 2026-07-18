<p align="center" >
    <img src="docs/logo.png"  width="60%" >
</p>

<h2 align="center">AnyRecon: Arbitrary-View 3D Reconstruction<br>with Video Diffusion Model</h2>

<div align="center">
  <a href="https://yutian10.github.io">Yutian Chen</a>, 
  <a href="https://guoshi28.github.io">Shi Guo</a>, 
  <a href="https://rbjin.github.io/">Renbiao Jin</a>,
  <a href="https://scholar.google.com/citations?user=9b5dE40AAAAJ&hl=en">Tianshuo Yang</a>,
  <a href="https://caixin98.github.io/">Xin Cai</a>, 
  <a href="https://luo0207.github.io/yawenluo/">Yawen Luo</a>, 
  <a href="">Mingxin Yang</a>, <br>
  <a href="https://mulinyu.github.io/">Mulin Yu</a>, 
  <a href="https://eveneveno.github.io/lnxu/">Linning Xu</a>, 
  <a href="https://tianfan.info/">Tianfan Xue</a>
</div>
<br>
<p align="center">
<a href="https://arxiv.org/pdf/2604.19747"><img src="https://img.shields.io/static/v1?label=Arxiv&message=AnyRecon&color=red&logo=arxiv"></a> &nbsp;
<a href="https://yutian10.github.io/AnyRecon/" target="_blank">
  <img src="https://img.shields.io/badge/Project%20Page-Website-3273dc?logo=googlechrome&logoColor=white">
</a> &nbsp;
 <a href='https://huggingface.co/Yutian10/AnyRecon'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow'></a> &nbsp;
 <a href='https://youtu.be/sfgFZKCdofs'><img src='https://img.shields.io/badge/YouTube-Video-FF0000?logo=youtube&logoColor=white'></a> &nbsp;
</p>

<h3 align="center"> SIGGRAPH Asia 2026 </h3>
<br>
<p align="center">
    <b>Your star means a lot for us to develop this project! ✨</b>
</p>
<p align="center" width="100%">
    <img src="docs/gif.gif"  width="90%" >
</p>


## TODO List

- [ ] Upload sparse attention weight.

## 🛠️ Environment Setup

###  1. Clone Repository and Setup Environment

The point-cloud rendering pipeline depends on [π³](https://github.com/yyfz/Pi3/), which is included as a git submodule. Make sure to clone **recursively** so that `Pi3/` is fetched at the same time:

```bash
git clone --recursive https://github.com/OpenImagingLab/AnyRecon.git
# If you already cloned without --recursive, run:
#   git submodule update --init --recursive
cd AnyRecon
conda create -n anyrecon python=3.10 -y
conda activate anyrecon
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r Pi3/requirements.txt
```

###  2. Download Models
AnyRecon relies on specific pre-trained weights. Please download them and place them in the `./checkpoints` folder.

- Base Video Diffusion Model (Wan2.1 I2V 14B 720P) [[download](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main)]
- AnyRecon LoRA weights [[download](https://huggingface.co/Yutian10/AnyRecon/tree/main)]
- π³ checkpoint (for point-cloud rendering) [[download](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors)] → place at `Pi3/model.safetensors`

## 🚀 Quick Start
For inference, processing an 869x512 video at 40 frames requires approximately 45GB of VRAM, but you can lower the resolution if your VRAM is insufficient. To reproduce the provided example, run:

```bash
bash test.sh
```

Or directly:

```bash
python run_AnyRecon.py \
    --root_dir example/valley \
    --output_dir example/valley \
    --lora_path full_attention.ckpt
```

## 🌟 Run on Your Own Data

`run_AnyRecon.py` expects point-cloud rendered **condition videos** as input. To prepare them from a raw video, we provide a helper script built on top of [π³](https://github.com/yyfz/Pi3/):

```bash
bash run_pi3.sh
```


**Input video format.** Your input video must be organized so that:

- the **first `--num_cond_frames` frames** are the **capture views** — these provide the 3D point cloud,
- the **remaining frames** are the **test views** — they are *only* used to estimate the camera poses at which the point cloud is rendered, and **do not contribute any points** to the reconstruction.


**Custom test-view trajectory (no test frames needed).** If you'd rather specify a custom rendering trajectory instead of estimating poses from real test-view images, you can replace the test-view portion of the video with any placeholder frames and override `target_extrinsics[num_cond_frames:]` inside `process_scene` with your desired sequence of `world→camera` 4×4 matrices. The capture views (the first `num_cond_frames` frames) will still be used to build the point cloud, and rendering proceeds along your chosen trajectory.

Once `run_pi3.py` has produced the condition videos in `--output_dir`, point `run_AnyRecon.py --root_dir` to that directory and run inference as shown above.

## 💗 Acknowledgments
Thanks to these great repositories: [Wan2.1](https://github.com/Wan-Video/Wan2.1), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), and [π³](https://github.com/yyfz/Pi3/).

## 🔗 Citation
If you find our work helpful, please cite it:
```
@article{chen2026anyrecon,
  title={AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model},
  author={Chen, Yutian and Guo, Shi and Jin, Renbiao and Yang, Tianshuo and Cai, Xin and Luo, Yawen and Yang, Mingxin and Yu, Mulin and Xu, Linning and Xue, Tianfan},
  journal={arXiv preprint arXiv:2604.19747},
  year={2026}
}
```
