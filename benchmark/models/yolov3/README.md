# YOLOV3
---
# Introduction
This is my own YOLOV3 written in pytorch, and is also the first time i have reproduced a object detection model.The dataset used is PASCAL VOC. The eval tool is the voc2010. Now the mAP gains the goal score.

Subsequently, i will continue to update the code to make it more concise , and add the new and efficient tricks.

`Note` : Now this repository supports the model compression in the new branch [model_compression](https://github.com/Peterisfar/YOLOV3/tree/model_compression)

---
## Results


| name | Train Dataset | Val Dataset | mAP(others) | mAP(mine) | notes |
| :----- | :----- | :------ | :----- | :-----| :-----|
| YOLOV3-448-544 | 2007trainval + 2012trainval | 2007test | 0.769 | 0.768 \| - | baseline(augument + step lr) |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.793 | 0.803 \| - | \+multi-scale training |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.806 | 0.811 \| - | \+focal loss(note the conf_loss in the start is lower) |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.808 | 0.813 \| - | \+giou loss |
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.812 | 0.821 \| - | \+label smooth |  
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.822 | 0.826 \| - | \+mixup |  
| YOLOV3-\*-544 | 2007trainval + 2012trainval | 2007test | 0.833 | 0.832 \| 0.840 | \+cosine lr |
| YOLOV3-\*-* | 2007trainval + 2012trainval | 2007test | 0.858 | 0.858 \| 0.860 | \+multi-scale test and flip, nms threshold is 0.45 |  

`Note` : 

* YOLOV3-448-544 means train image size is 448 and test image size is 544. `"*"` means the multi-scale.
* mAP(mine)'s format is (use_difficult mAP | no_difficult mAP).
* In the test, the nms threshold is 0.5(except the last one) and the conf_score is 0.01.`others` nms threshold is 0.45(0.45 will increase the mAP)
* Now only support the single gpu to train and test.


---
## Environment

* Nvida GeForce RTX 2080 Ti
* CUDA10.0
* CUDNN7.0
* ubuntu 16.04
* python 3.5
```bash
# install packages
pip3 install -r requirements.txt --user
```

---
## Brief

* [x] Data Augment (RandomHorizontalFlip, RandomCrop, RandomAffine, Resize)
* [x] Step lr Schedule 
* [x] Multi-scale Training (320 to 640)
* [x] focal loss
* [x] GIOU
* [x] Label smooth
* [x] Mixup
* [x] cosine lr
* [x] Multi-scale Test and Flip



---
## Prepared work

### 1、Git clone YOLOV3 repository
```Bash
git clone https://github.com/Peterisfar/YOLOV3.git
```
update the `"PROJECT_PATH"` in the params.py.
### 2、Download dataset
* Download Pascal VOC dataset : [VOC 2012_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 、[VOC 2007_trainval](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)、[VOC2007_test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar). put them in the dir, and update the `"DATA_PATH"` in the params.py.
* Convert data format : Convert the pascal voc *.xml format to custom format (Image_path0 &nbsp; xmin0,ymin0,xmax0,ymax0,class0 &nbsp; xmin1,ymin1...)

```bash
cd YOLOV3 && mkdir data
cd utils
python3 voc.py # get train_annotation.txt and test_annotation.txt in data/
```

### 3、Download weight file
* Darknet pre-trained weight :  [darknet53-448.weights](https://pjreddie.com/media/files/darknet53_448.weights) 
* This repository test weight : [best.pt](https://pan.baidu.com/s/1MdE2zfIND9NYd9mWytMX8g)

Make dir `weight/` in the YOLOV3 and put the weight file in.

---
## Train

Run the following command to start training and see the details in the `config/yolov3_config_voc.py`

```Bash
WEIGHT_PATH=weight/darknet53_448.weights

CUDA_VISIBLE_DEVICES=0 nohup python3 -u train.py --weight_path $WEIGHT_PATH --gpu_id 0 > nohup.log 2>&1 &

```

`Notes:`

* Training steps could run the `"cat nohup.log"` to print the log.
* It supports to resume training adding `--resume`, it will load `last.pt` automaticly.

---
## Test
You should define your weight file path `WEIGHT_FILE` and test data's path `DATA_TEST`
```Bash
WEIGHT_PATH=weight/best.pt
DATA_TEST=./data/test # your own images

CUDA_VISIBLE_DEVICES=0 python3 test.py --weight_path $WEIGHT_PATH --gpu_id 0 --visiual $DATA_TEST --eval

```
The images can be seen in the `data/`

---
## TODO

* [ ] Mish
* [ ] OctConv
* [ ] Custom data


---
## Reference

* tensorflow : https://github.com/Stinky-Tofu/Stronger-yolo
* pytorch : https://github.com/ultralytics/yolov3
* keras : https://github.com/qqwweee/keras-yolo3


