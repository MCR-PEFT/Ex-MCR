# Ex-MCR: Extending Multi-modal Contrastive Representation

Zehan Wang*, Ziang Zhang*, Luping Liu, Yang Zhao, Haifeng Huang, Tao Jin, Zhou Zhao*

Ex-MCR is a training-efficient and paired-data-free method to flexibly learn unified contrastive representation space, by integrating the knowledge of existing MCR spaces.

This implementation provides 3D-image-text-audio unified contrastive representation, obtained by aligning the representation spaces of CLAP (audio-text) and ULIP v2 (3D-vision) into the CLIP (vision-text).

![pipeline](./pipeline.png)

## News

- [09/22/2023] C-MCR has been accepted by NIPS 2023!🔥🔥🔥 [[paper](https://arxiv.org/abs/2305.14381)]
- [10/11/2023] Source Code of C-MCR has been released! [[code](https://github.com/MCR-PEFT/C-MCR)]


## File structure:
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

## Usage
### Install enviornments
Install pytorch 1.13+ and other 3rd party dependencies.
```shell
conda create -n exmcr python=3.8.16
conda activate exmcr
pip install -r requirements.txt
```

All feature extractors we use and their pretrained weights are shown below. You need to download the weights for CLAP and ULIP and put them in directory `checkpoints` and renamed them. The weights for CLIP will be downloaded automatically during the first running.

- **CLIP**：vit/B-32, you can find the repository [here](https://huggingface.co/openai/clip-vit-base-patch32)
- **CLAP**：LAION_AI/CLAP 630k-fusion-best.pt, you can find the repository [here](https://github.com/LAION-AI/CLAP) and you can download the weight we use [here](https://huggingface.co/lukewys/laion_clap/blob/main/630k-fusion-best.pt).
- **ULIP**：pointbert v2, you can find the repository [here](https://github.com/salesforce/ULIP) and you can download the weight we use [here](https://storage.cloud.google.com/sfr-ulip-code-release-research/pretrained_models/ckpt_zero-sho_classification/pointbert_ULIP-2.pt)


The final structure of `checkpoints` looks like this:
```
-checkpoints
	ex_clap.pt
	ex_ulip.pt
	laion_clap_fullset_fusion.pt
	pointbert_ULIP-2.pt
```

### Inference

Extract and compare embeddings in Base-MCR across modalities:
```python
from exmcr.exmcr_model import Ex_MCR
from exmcr.exmcr_model import ModalityType, MCRType
import torch

input = {ModalityType.VISION: ['assets/toilet.jpeg',
                               'assets/BBQ.jpeg',
                               'assets/train.jpeg'],
         ModalityType.TEXT: ['a toilet',
                             'BBQ',
                             'a train'],
         ModalityType.AUDIO:['assets/toilet.wav',
                             'assets/BBQ.wav',
                             'assets/train.wav'],
         ModalityType.PC:['assets/toilet.npy',
                          'assets/BBQ.npy',
                          'assets/train.npy']
         }

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Ex_MCR(device=device)

# you can get single modality embeddings by using these functions

# v_emb = model.get_vision_embedding(input)
# t_emb = model.get_text_embedding(input)
# a_emb = model.get_audio_embedding(input)
# p_emb = model.get_3d_embedding(input)

embeddings = model.get_embeddings_in_base_mcr(input)

print(
    "Vision x Text:\n",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T * 10.0, dim=-1)
)
print(
    "Audio x Text:\n",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T * 10.0, dim=-1)
)
print(
    "3D x VISION:\n",
    torch.softmax(embeddings[ModalityType.PC] @ embeddings[ModalityType.VISION].T * 10.0, dim=-1)
)
print(
    "Audio x Vision:\n",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.VISION].T * 10.0, dim=-1)
)
print(
    "3D x Text:\n",
    torch.softmax(embeddings[ModalityType.PC] @ embeddings[ModalityType.TEXT].T * 10.0, dim=-1)
)
print(
    "Audio x 3D:\n",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.PC].T * 10.0, dim=-1)
)

# Expected output

# Vision x Text:
#  tensor([[0.5737, 0.2242, 0.2021],
#         [0.2454, 0.4756, 0.2790],
#         [0.2541, 0.2530, 0.4929]], device='cuda:0')
# Audio x Text:
#  tensor([[0.5816, 0.2509, 0.1675],
#         [0.2025, 0.4903, 0.3072],
#         [0.2417, 0.2220, 0.5363]], device='cuda:0')
# 3D x VISION:
#  tensor([[0.6891, 0.1932, 0.1177],
#         [0.1549, 0.7470, 0.0981],
#         [0.0945, 0.0841, 0.8214]], device='cuda:0')
# Audio x Vision:
#  tensor([[0.5270, 0.2189, 0.2541],
#         [0.1978, 0.7140, 0.0881],
#         [0.1574, 0.0984, 0.7442]], device='cuda:0')
# 3D x Text:
#  tensor([[0.7076, 0.1632, 0.1292],
#         [0.1881, 0.5662, 0.2457],
#         [0.1429, 0.2350, 0.6221]], device='cuda:0')
# Audio x 3D:
#  tensor([[0.8921, 0.0792, 0.0286],
#         [0.0729, 0.8420, 0.0851],
#         [0.0361, 0.0315, 0.9324]], device='cuda:0')
```

## TODO

- [x] Install environments
- [x] News
- [x] brief introduction for Ex-MCR
- [x] Usage
- [x] Citation
- [ ] Code for training
- [x] Code for inference


## Citation
If you find this proiect useful in our research, please consider giving a star :star: and citation:
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

lf you have any questions or suggestions, feel free to drop us an email ( wangzehan01@zju.edu.cn, ziangzhang@zju.edu.cn ) or open an issue.

### Acknowledgement 
Thanks to the open source of the following projects:
[CLIP](https://huggingface.co/openai/clip-vit-base-patch32), [CLAP](https://github.com/LAION-AI/CLAP), [ULIP](https://github.com/salesforce/ULIP), [Imagebind](https://github.com/facebookresearch/ImageBind).