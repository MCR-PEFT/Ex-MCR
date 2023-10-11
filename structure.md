```
-assets
	[demo samples, including image, audio and 3d model]
-checkpoints
	[weight of clip,clap,ulip and ex-mcr_clap, ex-mcr_ulip]
-exmcr
	-models
		- ULIP [source code of ULIP]
		exmcr_projector.py [the projector of ex-mcr]
		exmcr_trunks.py [feature extractor of clip, clap and ulip]
		exmcr_model.py [combine projector and trunks together with useful functions]
		type.py
		
```

clip：vit/B-32

clap：laion_clap enable_fusion=True

ulip：pointbert v2


	