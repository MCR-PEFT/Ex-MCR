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