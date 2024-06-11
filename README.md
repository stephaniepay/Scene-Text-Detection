## Requirements:
- PyTorch 1.13.1 
- CUDA 11.6
- GPU-P6000 24GB, 30GB RAM, and 8 CPUs


## File naming description:

#### Stage 1: Backbone Switching
- Run the ipynb files
- Each file represents a model

#### Stage 2: Proposed Model Implementation
- Run the ipynb files
- Each file represent the proposed model with Spatial Transformer Network, Atrous Spatial Pyramid Pooling, and ResNet50


## Folder Description:

data: Total-Text (dataset used for training and testing)
<br>dataset: Codes to run the data
<br>Models: Weights of all models
<br>output: Output pixel coordinates folder
<br>util: Codes for configuration and post-processing
<br>vis: Output images visualization folder 


## Folders in vis and output description:

resnet_ASPP_LATE_3: 
Stage 2 - RESNET50 +ASPP

resnet_ASPP_LATE_GATE: 
Stage 2 - RESNET50 +STN +GATE + ASPP

resnet_ASPP_LATE_2: 
Stage 2 - RESNET50 +STN + ASPP

resnet_STN_LATE: 
Stage 2 - RESNET50 +STN

vgg123456: 
Stage 1 - VGG16

resnet_original_2: 
Stage 1 - ResNet50

densenet_original: 
Stage 1 - DenseNet121



