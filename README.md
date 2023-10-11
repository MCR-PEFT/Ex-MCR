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

CLIP：vit/B-32, you can find the repository ![here](https://huggingface.co/openai/clip-vit-base-patch32)

CLAP：LAION_AI/CLAP 630k-fusion-best.pt, you can find the repository ![here](https://github.com/LAION-AI/CLAP) and you can download the weight we use ![here](https://huggingface.co/lukewys/laion_clap/blob/main/630k-fusion-best.pt).

ULIP：pointbert v2, you can find the repository ![here](https://github.com/salesforce/ULIP) and you can download the weight we use ![here](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/pointbert_ULIP-2.pt)


### Pretrained Weights of Projectors
Ex-CLAP: ![Google_Drive](https://drive.google.com/file/d/19GNAZi_A7Zqb8ZfDkvo4yIpKinQ-1Sme/view?usp=sharing)
Ex-ULIP: ![Google_Drive](https://drive.google.com/file/d/16QtRCn3U-kfU_xtE0mdYp0fFznJc59F3/view?usp=sharing)

### Install enviornments

```shell
conda create -n exmcr python=3.8.16
conda activate exmcr
pip install -r requirements.txt
```

For inferencing, you should put pretrained weights in directory `checkpoints`, as below:
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
- [ ] What is Ex-MCR
- [ ] How to Use
- [ ] License
- [ ] Citation



