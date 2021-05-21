from torch.utils.data import DataLoader
import utils.gpu as gpu
from model.yolov3 import Yolov3
from tqdm import tqdm
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
import os
import config.yolov3_config_voc as cfg
from utils.visualize import *

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'


class Tester(object):
    def __init__(self,
                 weight_path=None,
                 gpu_id=0,
                 img_size=544,
                 eval=False
                 ):
        self.img_size = img_size
        self.__num_class = cfg.DATA["NUM"]
        self.__conf_threshold = cfg.TEST["CONF_THRESH"]
        self.__nms_threshold = cfg.TEST["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__multi_scale_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.__flip_test = cfg.TEST["FLIP_TEST"]

        self.__eval = eval
        self.__classes = cfg.DATA["CLASSES"]

        self.__model = Yolov3().to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model)


    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt


    def test(self):
        if self.__eval:
            mAP = 0
            print('*' * 20 + "Validate" + '*' * 20)

            with torch.no_grad():
                APs = Evaluator(self.__model).APs_voc()

                for i in APs:
                    print("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self.__num_class
                print('mAP:%g' % (mAP))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/best.pt', help='weight file path')
    parser.add_argument('--eval', action='store_true', default=True, help='eval the mAP or not')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    opt = parser.parse_args()

    Tester( weight_path=opt.weight_path,
            gpu_id=opt.gpu_id,
            eval=opt.eval).test()
