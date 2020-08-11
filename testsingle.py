import os
import time
import logging

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms
from util import AverageMeter

cv2.ocl.setUseOpenCL(False)


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global logger
    logger = get_logger()

    result_folder = os.path.join("result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    data_list = os.listdir("testJPEGImages")

    from torch2trt import TRTModule
    model = TRTModule()
    model.load_state_dict(torch.load("bisenet_fp16.pth"))
    model = model.cuda().half()
    test(data_list, model, result_folder)



def test(data_list, model, result_folder):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    step_time = AverageMeter()
    model.eval()

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    for i,img_path in enumerate(data_list):
        img_path = os.path.join("testJPEGImages",img_path)
        image = Image.open(img_path).convert('RGB')
        image = image_transform(image).cuda()
        input = torch.unsqueeze(image,0)

        with torch.no_grad():

            start_time = time.time()
            prediction = model(input.half())
            out = (prediction>0.9).int()
            out = out.squeeze().cpu().numpy()

            end_time = time.time()
            per_time = end_time - start_time

        step_time.update(per_time)

        if ((i + 1) % 10 == 0) or (i + 1 == len(data_list)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {step_time.val:.3f} ({step_time.avg:.3f}).'.format(i + 1, len(data_list),
                                                                                    data_time=data_time,
                                                                                    step_time=step_time))
        gray = np.uint8(out)
        gray = gray*255
        image_path = data_list[i]
        result_path = os.path.join(result_folder, image_path)
        cv2.imwrite(result_path, gray)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')



if __name__ == '__main__':
    main()
