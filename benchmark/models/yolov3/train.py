import logging
import utils.gpu as gpu
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
import argparse
from eval.evaluator import *
from utils.tools import *
import config.yolov3_config_voc as cfg
from utils import cosine_lr_scheduler

import adaptdl
import adaptdl.env
import adaptdl.torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from apex import amp
from apex.amp._amp_state import _amp_state

from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics, get_progress


class Trainer(object):
    def __init__(self, weight_path):
        init_seeds(0)
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = adaptdl.torch.AdaptiveDataLoader(self.train_dataset,
                                                                 batch_size=cfg.TRAIN["BATCH_SIZE"],
                                                                 num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                                                 drop_last=True,
                                                                 shuffle=True)
        self.train_dataloader.autoscale_batch_size(512, local_bsz_bounds=(4, 8),
                                                   gradient_accumulation=True)
        self.valid_dataset = data.VocDataset(anno_file_type="test")
        self.valid_dataloader = adaptdl.torch.AdaptiveDataLoader(self.valid_dataset,
                                                                 batch_size=(8 * adaptdl.env.num_replicas()),
                                                                 num_workers=8,
                                                                 shuffle=False)
        self.yolov3 = Yolov3().cuda()

        self.optimizer = optim.SGD(self.yolov3.parameters(), lr=cfg.TRAIN["LR_INIT"],
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.yolov3.load_darknet_weights(weight_path)

        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=(self.epochs - cfg.TRAIN["WARMUP_EPOCHS"]),
                                           eta_min=cfg.TRAIN["LR_END"])
        adaptdl.torch.init_process_group("nccl")
        self.yolov3, self.optimizer = amp.initialize(self.yolov3, self.optimizer)
        self.yolov3 = adaptdl.torch.AdaptiveDataParallel(self.yolov3, self.optimizer, self.scheduler,
                                                         patch_optimizer=False)
        self.yolov3.adascale._smoothing = 0.997

    def valid(self, epoch):
        self.yolov3.train()

        accum = adaptdl.torch.Accumulator()
        with torch.no_grad():
            for imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes in self.valid_dataloader:
                imgs = imgs.cuda()
                label_sbbox = label_sbbox.cuda()
                label_mbbox = label_mbbox.cuda()
                label_lbbox = label_lbbox.cuda()
                sbboxes = sbboxes.cuda()
                mbboxes = mbboxes.cuda()
                lbboxes = lbboxes.cuda()

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(
                        p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)

                accum["loss_sum"] += loss.item() * imgs.size(0)
                accum["loss_cnt"] += imgs.size(0)

                # Print batch results
                print("Epoch {} valid [{}/{}]:  loss_giou: {:.4f}  loss_conf: {:.4f}  loss_cls: {:.4f}  loss: {:.4f}"
                      .format(epoch, self.valid_dataloader._elastic.current_index, len(self.valid_dataset),
                              loss_giou.item(), loss_conf.item(), loss_cls.item(), loss.item()))

                del imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
                del p, p_d, loss, loss_giou, loss_conf, loss_cls

        with accum.synchronized(), SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
            accum["loss_avg"] = accum["loss_sum"] / accum["loss_cnt"]
            writer.add_scalar("Loss/Valid", accum["loss_avg"], epoch)
            report_valid_metrics(epoch, accum["loss_avg"])
            print("Valid:", accum)


    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))
        for epoch in adaptdl.torch.remaining_epochs_until(self.epochs):
            self.yolov3.train()
            accum = adaptdl.torch.Accumulator()
            with SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
                for imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes in self.train_dataloader:
                    imgs = imgs.cuda()
                    label_sbbox = label_sbbox.cuda()
                    label_mbbox = label_mbbox.cuda()
                    label_lbbox = label_lbbox.cuda()
                    sbboxes = sbboxes.cuda()
                    mbboxes = mbboxes.cuda()
                    lbboxes = lbboxes.cuda()

                    p, p_d = self.yolov3(imgs)

                    loss, loss_giou, loss_conf, loss_cls = self.criterion(
                            p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)

                    delay_unscale = not self.train_dataloader._elastic.is_sync_step()
                    with amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:
                        self.yolov3.adascale.loss_scale = _amp_state.loss_scalers[0].loss_scale()
                        scaled_loss.backward()
                    self.yolov3.adascale.step()

                    accum["loss_sum"] += loss.item() * imgs.size(0)
                    accum["loss_cnt"] += imgs.size(0)

                    # Print batch results
                    print("Epoch {} train [{}/{}]:  img_size: {}  loss_giou: {:.4f}  loss_conf: {:.4f}  loss_cls: {:.4f}  loss: {:.4f}"
                          .format(epoch, self.train_dataloader._elastic.current_index, len(self.train_dataset),
                                  self.train_dataset.img_size, loss_giou.item(), loss_conf.item(), loss_cls.item(), loss.item()))

                    # Multi-scale training (320-608 pixels).
                    if self.multi_scale_train:
                        self.train_dataset.img_size = random.choice(range(10, 20)) * 32

                    global_step = int(self.yolov3.adascale._state["progress"])
                    self.train_dataloader.to_tensorboard(writer, global_step, tag_prefix="AdaptDL/Data")
                    self.yolov3.to_tensorboard(writer, global_step, tag_prefix="AdaptDL/Model")

                    del imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
                    del p, p_d, loss, loss_giou, loss_conf, loss_cls

                    if epoch < cfg.TRAIN["WARMUP_EPOCHS"]:
                        for group in self.optimizer.param_groups:
                            group["lr"] = (get_progress() / len(self.train_dataset) *
                                           self.train_dataloader.batch_size /
                                           cfg.TRAIN["WARMUP_EPOCHS"]) * cfg.TRAIN["LR_INIT"]
                    print("lr =", self.optimizer.param_groups[0]["lr"])

                with accum.synchronized():
                    accum["loss_avg"] = accum["loss_sum"] / accum["loss_cnt"]
                    writer.add_scalar("Loss/Train", accum["loss_avg"], epoch)
                    report_train_metrics(epoch, accum["loss_avg"])
                    print("Train:", accum)

            self.valid(epoch)

            if epoch >= cfg.TRAIN["WARMUP_EPOCHS"]:
                self.scheduler.step()

        #with torch.no_grad(), SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
        #    print('*'*20+"Evaluate"+'*'*20)
        #    APs = Evaluator(self.yolov3).APs_voc()
        #    mAP = 0
        #    for i in APs:
        #        print("{} --> mAP : {}".format(i, APs[i]))
        #        mAP += APs[i]
        #    mAP = mAP / self.train_dataset.num_classes
        #    print('mAP:%g'%(mAP))
        #    writer.add_scalar("Eval/mAP", float(mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    opt = parser.parse_args()

    Trainer(weight_path=opt.weight_path).train()
