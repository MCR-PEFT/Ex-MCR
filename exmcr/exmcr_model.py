import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from exmcr.exmcr_projector import Ex_MCR_Head
from exmcr.exmcr_trunks import Trunk

from exmcr.type import ModalityType, MCRType

EX_CLAP = 'checkpoints/ex_clap.pt'
EX_ULIP = 'checkpoints/ex_ulip.pt'

class Ex_MCR():
    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.ex_mcr_heads = self._create_heads(device)
        
        self.trunk = Trunk(device) # dict
    
    def _create_heads(self, device):
        heads = {}
        heads[MCRType.CLAP] = Ex_MCR_Head()
        heads[MCRType.ULIP] = Ex_MCR_Head()
        heads[MCRType.CLAP].load_state_dict(torch.load(EX_CLAP, map_location='cpu'))
        heads[MCRType.ULIP].load_state_dict(torch.load(EX_ULIP, map_location='cpu'))
        heads[MCRType.CLAP].to(device)
        heads[MCRType.ULIP].to(device)
        return heads
    
    @torch.no_grad()
    def _get_features_from_input(self, input: dict) -> dict:
        return self.trunk.extract_feature_from_input(input)
    
    @torch.no_grad()
    def project_features_to_base_mcr(self, features: dict) -> dict:
        base_embeddings = {}
        base_embeddings[ModalityType.VISION] = features[ModalityType.VISION]
        base_embeddings[ModalityType.TEXT]   = features[ModalityType.TEXT]
        # project audio embedding to base-MCR
        base_embeddings[ModalityType.AUDIO]  = self.project_audio_to_base_mcr(features[ModalityType.AUDIO])
        # project pointcloud embedding to base-MCR
        base_embeddings[ModalityType.PC]     = self.project_3d_to_base_mcr(features[ModalityType.PC])
        
        
        base_embeddings[ModalityType.VISION] = F.normalize(base_embeddings[ModalityType.VISION], dim=-1)
        base_embeddings[ModalityType.TEXT]   = F.normalize(base_embeddings[ModalityType.TEXT], dim=-1)
        base_embeddings[ModalityType.AUDIO]  = F.normalize(base_embeddings[ModalityType.AUDIO], dim=-1)
        base_embeddings[ModalityType.PC]     = F.normalize(base_embeddings[ModalityType.PC], dim=-1)
        
        return base_embeddings
    
    @torch.no_grad()
    def project_audio_to_base_mcr(self, audio_emb: Tensor) -> Tensor:
        # project audio embedding to base-MCR
        audio_emb = self.ex_mcr_heads[MCRType.CLAP].Head1(audio_emb)
        audio_emb = self.ex_mcr_heads[MCRType.CLAP].Head2(audio_emb)
        audio_emb = audio_emb
        return audio_emb
    
    @torch.no_grad()
    def project_3d_to_base_mcr(self, point_emb: Tensor) -> Tensor:
        # project pointcloud embedding to base-MCR
        point_emb = self.ex_mcr_heads[MCRType.ULIP].Head1(point_emb)
        point_emb = self.ex_mcr_heads[MCRType.ULIP].Head2(point_emb)
        point_emb = point_emb
        
        return point_emb
    
    @torch.no_grad()
    def get_embeddings_in_base_mcr(self, input: dict) -> dict:
        features = self._get_features_from_input(input)
        base_embeddings = self.project_features_to_base_mcr(features)
        return base_embeddings
    
    @torch.no_grad()
    def get_vision_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_vision_feature(input[ModalityType.VISION])
        return F.normalize(features, dim=-1)
    
    @torch.no_grad()
    def get_text_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_text_feature(input[ModalityType.TEXT])
        return F.normalize(features, dim=-1)
    
    @torch.no_grad()
    def get_audio_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_audio_feature(input[ModalityType.AUDIO])
        base_embeddings = self.project_audio_to_base_mcr(features)
        return F.normalize(base_embeddings, dim=-1)
    
    @torch.no_grad()
    def get_3d_embedding(self, input: dict) -> Tensor:
        features = self.trunk.get_3d_feature(input[ModalityType.PC])
        base_embeddings = self.project_3d_to_base_mcr(features)
        return F.normalize(base_embeddings, dim=-1)
    