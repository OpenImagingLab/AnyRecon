<p align="center" >
    <img src="docs/logo.png"  width="60%" >
</p>

<h2 align="center">AnyRecon: Arbitrary-View 3D Reconstruction<br>with Video Diffusion Model</h2>

<br>

<p align="center" width="100%">
    <img src="docs/gif.gif"  width="90%" >
</p>

## TODO List

- [ ] Upload sparse attention weight.

## 🛠️ Environment Setup

###  1. Clone Repository and Setup Environment
```bash
git clone https://github.com/OpenImagingLab/AnyRecon.git
cd AnyRecon
conda create -n anyrecon python=3.10 -y
conda activate anyrecon
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

###  2. Download Models
AnyRecon relies on specific pre-trained weights. Please download them and place them in the `./checkpoints` folder.

- Base Video Diffusion Model (Wan2.1 I2V 14B 720P) [[download](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main)]
- AnyRecon LoRA weights [[download](https://huggingface.co/Yutian10/AnyRecon/tree/main)]

## 🚀 Quick Start

### Inference
You can run the inference using the provided `test.sh` script:

```bash
bash test.sh
```

Or you can run the python script directly:
```bash
python run_AnyRecon.py \
    --root_dir example/valley \
    --output_dir example/valley \
    --lora_path full_attention.ckpt
```

## 💗 Acknowledgments
Thanks to these great repositories: [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

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
