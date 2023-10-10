from exmcr.exmcr_model import Ex_MCR
from exmcr.exmcr_model import ModalityType, MCRType

input = {ModalityType.VISION: ['assets/c8b68cf87c994eeb99e8bbcc5ce84c4e.jpeg',
                               'assets/b21b9753332d4e46a3deee2b0151bd8e.jpeg',
                               'assets/0b76d30d697d4488bd35dee187d95fac.jpeg'],
         ModalityType.TEXT: ['Someone is using the toilet.',
                             'The dog snarled at us.',
                             'The helicopter is circling overhead.'],
         ModalityType.AUDIO:['assets/AagLJkfrFMk.wav',
                             'assets/0yxEvdnimGg.wav',
                             'assets/KvrcRMfFzOE.wav'],
         ModalityType.PC:['assets/b3d1ab7790954d18a46ad83c8c0c3594.npy',
                          'assets/78d6a35f0ecf4cf69ae9271f28f5f137.npy',
                          'assets/c2dca5937bf6488ea938ea9a6dbe96ba.npy']
         }

model = Ex_MCR(device='cuda:3')

# v_emb = model.get_vision_embedding(input)
# t_emb = model.get_text_embedding(input)
# a_emb = model.get_audio_embedding(input)
# p_emb = model.get_3d_embedding(input)

output = model.get_embeddings_in_base_mcr(input)

v_emb = output[ModalityType.VISION]
t_emb = output[ModalityType.TEXT]
a_emb = output[ModalityType.AUDIO]
p_emb = output[ModalityType.PC]

sim = a_emb @ v_emb.T
print(sim)
logits = sim.argmax(dim=-1)
print(logits)

sim = p_emb @ t_emb.T
print(sim)
logits = sim.argmax(dim=-1)
print(logits)

sim = a_emb @ p_emb.T
print(sim)
logits = sim.argmax(dim=-1)
print(logits)

print('over')