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
#         [0.2541, 0.2530, 0.4929]], device='cuda:3')
# Audio x Text:
#  tensor([[0.9115, 0.0369, 0.0516],
#         [0.1443, 0.7265, 0.1292],
#         [0.1000, 0.0960, 0.8040]], device='cuda:3')
# 3D x VISION:
#  tensor([[0.8635, 0.0850, 0.0515],
#         [0.0761, 0.9125, 0.0114],
#         [0.1239, 0.0781, 0.7980]], device='cuda:3')
# Audio x Vision:
#  tensor([[0.9060, 0.0329, 0.0611],
#         [0.0743, 0.8742, 0.0515],
#         [0.0648, 0.0227, 0.9126]], device='cuda:3')
# 3D x Text:
#  tensor([[0.8052, 0.0909, 0.1039],
#         [0.2447, 0.6035, 0.1518],
#         [0.1757, 0.2094, 0.6149]], device='cuda:3')
# Audio x 3D:
#  tensor([[0.9860, 0.0038, 0.0102],
#         [0.0078, 0.9875, 0.0048],
#         [0.0204, 0.0021, 0.9775]], device='cuda:3')