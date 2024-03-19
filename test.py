
import os
import sys
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.PIMNet import PIMNet
from thop import profile
from dataload.data_util import Config,Data
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__=='__main__':
    # model_pth = "./pths/splice27k_1000_lr0.05_size384_att_sgd/30.pth"
    model_pth = "./pths/"
    test_paths = "./data/"

    cfg = Config(datapath=test_paths, snapshot=model_pth, mode='test')
    data = Data(cfg)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    net = PIMNet(cfg)

    net.train(False)
    net.cuda()
    dataset_name = test_paths.split('/')[-1]
    with torch.no_grad():
        test_time = AverageMeter()
        input = torch.randn(1, 3, 384, 384).cuda()
        flops, params = profile(net, inputs=([input]))
        print(str(flops / 1e9) + 'G')
        print(str(params / 1e6) + 'M')
        for image, mask, shape, name in tqdm(loader):
            image = image.cuda().float()
            begin = time.time()
            pred = net(image, shape)
            test_time.update(time.time() - begin)
            pred = (torch.sigmoid(pred[0, 0]) * 255).cpu().numpy()
            directory = './results/' + model_pth.split('/')[-2].replace('.pth','') +'/'+ dataset_name
            if not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(directory + '/' + name[0] + '.png', np.round(pred))
        fps = 1 / test_time.avg
        print("count:", test_time.count, "test_time:", test_time.avg, "fps:", fps)
