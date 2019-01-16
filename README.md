# SlowFastNetworks
PyTorch implementation of ["SlowFast Networks for Video Recognition"](https://arxiv.org/abs/1812.03982).
## Train
1. Dataset should be orgnized as：  
```
dataset(e.g. UCF-101)  
│    │ train/training  
│    │    │ ApplyEyeMakeup  
│    │    │ ApplyLipstick  
│    │    │ ...  
│    │ validation  
     │    │ ApplyEyeMakeup  
     │    │ ApplyLipstick  
     │    │ ...   
```

2. Modify the params in config.py and `mode` of `train_dataloader` or `val_dataloader` in train.py.   

## Requirements
python 3  
PyTorch >= 0.4.1  
tensorboardX  
OpenCV  

## Code Reference:
[1] https://github.com/Guocode/SlowFast-Networks/  
[2] https://github.com/jfzhang95/pytorch-video-recognition  
[3] https://github.com/irhumshafkat/R2Plus1D-PyTorch  
