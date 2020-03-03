# RP-R-CNN
Official implementation of **Renovating Parsing R-CNN for Accurate Multiple Human Parsing (under review)**

In this repository, we release the RP R-CNN code in Pytorch.

- RP R-CNN architecture:
<p align="center"><img width="90%" src="data/rp_rcnn.png" /></p>

- RP R-CNN output:
<p align="center"><img width="75%" src="data/output.png" /></p>


## Installation
- 8 x TITAN Xp GPU
- pytorch1.1
- python3.6.8

Install RP R-CNN following [INSTALL.md](https://github.com/soeaver/RP-R-CNN/blob/master/INSTALL.md#install).


## Results and Models

**On CIHP**

|  Backbone  |  LR  | Det AP | Parsing (mIoU/APp50/APvol/PCP50) | DOWNLOAD |
|------------|:----:|:------:|:--------------------------------:| :-------:|
|  R-50-FPN  |  3x  | 67.3   |        58.2/71.6/58.3/62.2       | [GoogleDrive](https://drive.google.com/open?id=1-nOef31NrjMyZXkK8fJRmPi-aRQJm7dS)|
|  R-50-FPN  |  6x  | 68.2   |        60.2/74.1/59.5/64.9       | [GoogleDrive](https://drive.google.com/open?id=1-nOef31NrjMyZXkK8fJRmPi-aRQJm7dS)|
|    +tta    |  6x  | 68.2   |        60.2/74.1/59.5/64.9       | [GoogleDrive](https://drive.google.com/open?id=1-nOef31NrjMyZXkK8fJRmPi-aRQJm7dS)|

**ImageNet pretrained weight**

- [R-50](https://drive.google.com/open?id=1EtqFhrFTdBJNbp67effArVrTNx4q_ELr)
- [X-101-32x8d](https://drive.google.com/open?id=1c4OSVZIZtDT49B0DTC0tK3vcRgJpzR9n)


## Training

To train a model with 8 GPUs run:
```
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --cfg cfgs/mscoco_humanparts/e2e_hier_rcnn_R-50-FPN_1x.yaml
```


## Evaluation

### multi-gpu evaluation,
```
python tools/test_net.py --cfg ckpts/mscoco_humanparts/e2e_hier_rcnn_R-50-FPN_1x/e2e_hier_rcnn_R-50-FPN_1x.yaml --gpu_id 0,1,2,3,4,5,6,7
```

### single-gpu evaluation,
```
python tools/test_net.py --cfg ckpts/mscoco_humanparts/e2e_hier_rcnn_R-50-FPN_1x/e2e_hier_rcnn_R-50-FPN_1x.yaml --gpu_id 0
```


## License
Hier-R-CNN is released under the [MIT license](https://github.com/soeaver/Hier-R-CNN/blob/master/LICENSE).
