# SalEMA-multi-expert-on-CMP
Create two versions of an existed model for dynamic saliency prediction on 2D video, new models are designed to predict saliency on 360 videos , one trained to predict over the equator and one trained to predict over poles of dynamic omnidirectional stimuli ,
Changes are applied on SalEMA architecture in order to adapt 2D projected paches characteristics .
- to project from ERP to CMP format
```shell
python project_to_cube.py
```
- to project from CMP to ERP format
```shell
python project_to_equi.py
```
- requirements
```shell
pip install matplotlib
pip install numpy
pip install opencv-python
pip install Pillow
pip install torch
pip install torchvision
```

# Reference 
SalEMA from "Simple vs complex temporal recurrences for video saliency prediction (BMVC 2019)"

SalEMA is a video saliency prediction network. It utilizes a moving average of convolutional states to produce state of the art results. The architecture has been trained on DHF1K.
- https://imatge-upc.github.io/SalEMA/
- https://github.com/Linardos/SalEMA
