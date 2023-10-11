from exmcr.exmcr_model import Ex_MCR
from exmcr.exmcr_model import ModalityType, MCRType
import torch

input = {ModalityType.VISION: ['assets/toilet.jpeg',
                               'assets/dog.jpeg',
                               'assets/helicopter.jpeg'],
         ModalityType.TEXT: ['Someone is using the toilet.',
                             'The dog snarled at us.',
                             'The helicopter is circling overhead.'],
         ModalityType.AUDIO:['assets/toilet.wav',
                             'assets/dog.wav',
                             'assets/helicopter.wav'],
         ModalityType.PC:['assets/toilet.npy',
                          'assets/dog.npy',
                          'assets/helicopter.npy']
         }

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Ex_MCR(device=device)

# you can get single modality embeddings by using these functions

# v_emb = model.get_vision_embedding(input)
# t_emb = model.get_text_embedding(input)
# a_emb = model.get_audio_embedding(input)
# p_emb = model.get_3d_embedding(input)

output = model.get_embeddings_in_base_mcr(input)

v_emb = output[ModalityType.VISION]
t_emb = output[ModalityType.TEXT]
a_emb = output[ModalityType.AUDIO]
p_emb = output[ModalityType.PC]

print('Audio-Vision Results')
sim = a_emb @ v_emb.T
print(sim)
logits = sim.argmax(dim=-1)
print(logits)

print('3D-Text Results')
sim = p_emb @ t_emb.T
print(sim)
logits = sim.argmax(dim=-1)
print(logits)

print('Audio-3D Results')
sim = a_emb @ p_emb.T
print(sim)
logits = sim.argmax(dim=-1)
print(logits)

print('over')