```
-assets
	[示例数据样本，包括图片、音频和3D模型]
-checkpoints
	[clip,clap,ulip的权重文件以及ex-mcr_clap, ex-mcr_ulip的权重文件]
-exmcr
	-models
		- ULIP [ULIP 代码]
		exmcr_projector.py [ex-mcr的投影器]
		exmcr_trunks.py [对应模态的特征提取代码，包括clip，clap和ulip]
		exmcr_model.py [封装trunk和projector为一个类，以及权重初始化]
	data.py [文件名作为输入的数据读取以及预处理函数]
		
```

clip：vit/B-32

clap：laion_clap enable_fusion=True

ulip：pointbert v2


	