# Ex-MCR

### Framework

todo

### file structure:
```
-assets
	[demo samples, including image, audio and 3d model]
-checkpoints
	[weight of clip,clap,ulip and ex-mcr_clap, ex-mcr_ulip]
-exmcr
	- ULIP [source code of ULIP]
	exmcr_projector.py [the projector of ex-mcr]
	exmcr_trunks.py [feature extractor of clip, clap and ulip]
	exmcr_model.py [combine projector and trunks together with useful functions]
	type.py
		
```

### Feature Extractor
feature extractor we use:

CLIP：vit/B-32, you can find the repository [here](https://huggingface.co/openai/clip-vit-base-patch32)

CLAP：LAION_AI/CLAP 630k-fusion-best.pt, you can find the repository [here](https://github.com/LAION-AI/CLAP) and you can download the weight we use [here](https://huggingface.co/lukewys/laion_clap/blob/main/630k-fusion-best.pt).

ULIP：pointbert v2, you can find the repository [here](https://github.com/salesforce/ULIP) and you can download the weight we use [here](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/pointbert_ULIP-2.pt)


### Pretrained Weights for Projectors
Ex-CLAP: [Google_Drive](https://drive.google.com/file/d/19GNAZi_A7Zqb8ZfDkvo4yIpKinQ-1Sme/view?usp=sharing)
Ex-ULIP: [Google_Drive](https://drive.google.com/file/d/16QtRCn3U-kfU_xtE0mdYp0fFznJc59F3/view?usp=sharing)

### Install enviornments

```shell
conda create -n exmcr python=3.8.16
conda activate exmcr
pip install -r requirements.txt
```

For inferencing, you need to create the `checkpoints` diretory and put pretrained weights in it, as below:
```
-checkpoints
	ex_clap.pt
	ex_ulip.pt
	laion_clap_fullset_fusion.pt
	pointbert_ULIP-2.pt
```
Then you can test the model:
```shell
python inference.py
```

### TODO

- [x] Install environments(CLIP, CLAP, ULIP, KNN, librosa, torch, torchvision, torchaudio, numpy, Ninja)
- [ ] News
- [ ] What is Ex-MCR
- [x] How to Use
- [x] Citation


### Citation
If you find this proiect useful in vour research, please consider cite:
```
@misc{wang2023connecting,
      title={Connecting Multi-modal Contrastive Representations}, 
      author={Zehan Wang and Yang Zhao and Xize Cheng and Haifeng Huang and Jiageng Liu and Li Tang and Linjun Li and Yongqi Wang and Aoxiong Yin and Ziang Zhang and Zhou Zhao},
      year={2023},
      eprint={2305.14381},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

lf you have any questions or suggestions, feel free to drop us an email ( wangzehan01@zju.edu.cn, ziangzhang_vsama@qq.com )or open an issue.

### Acknowledgement 
Thanks to the open source of the following projects:[CLIP](https://huggingface.co/openai/clip-vit-base-patch32), [CLAP](https://github.com/LAION-AI/CLAP), [ULIP](https://github.com/salesforce/ULIP), [Imagebind](https://github.com/facebookresearch/ImageBind).