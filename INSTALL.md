## Install

```
# install pytorch 1.4 and torchvision
sudo pip3 install torch==1.4.0 torchvision

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
sudo python setup.py install --cuda_ext --cpp_ext

# clone RP-R-CNN
git clone https://github.com/soeaver/RP-R-CNN.git

# install other requirements
pip3 install -r requirements.txt

# mask ops
cd RP-R-CNN
sh make.sh

# make cocoapi
cd RP-R-CNN/cocoapi/PythonAPI
mask
cd ../../
ln -s cocoapi/PythonAPI/pycocotools/ ./
```

## Data and Pre-train weights

  Make sure to put the files as the following structure:

  ```
├─data
│  ├─CIHP
│  │  ├─train_img
│  │  │─train_parsing
│  │  │─train_seg
│  │  ├─val_img
│  │  │─val_parsing
│  │  │─val_seg  
│  │  ├─annotations
│  │  │  ├─CIHP_train.json
│  │  │  ├─CIHP_val.json
|  |
│  ├─MHP-v2
│  │  ├─train_img
│  │  │─train_parsing
│  │  │─train_seg
│  │  ├─val_img
│  │  │─val_parsing
│  │  │─val_seg  
│  │  ├─annotations
│  │  │  ├─MHP-v2_train.json
│  │  │  ├─MHP-v2_val.json
|
├─weights
   ├─resnet50_caffe.pth
   ├─resnet101_caffe.pth
   ├─resnext101_32x8d-8ba56ff5.pth

  ```
  

