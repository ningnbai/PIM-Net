import os
import sys
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from models.PIMNet import PIMNet
from tensorboardX import SummaryWriter
from dataload.data_util import Config, Data
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


if __name__ == '__main__':
    nets_name = 'polyp'
    cfg = Config(datapath='./data/',
                 savepath='./pths/', mode='train',
                 batch=16, lr=1e-4, momen=0.9, decay=5e-4, epoch=30, lr_decay_gamma=0.1)

    data = Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=0)

    net = PIMNet(cfg)
    save_tensorboard_dir = './tensorboard_server/' + nets_name + '/'
    if not os.path.exists(save_tensorboard_dir):
        os.makedirs(save_tensorboard_dir)
    save_pth_dir = './pths/' + nets_name + '/'
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    net.train(True)
    net.cuda()

    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.Adam([{'params': base}, {'params': head}], lr=cfg.lr, betas=(0.9, 0.999),
                                  weight_decay=cfg.decay)

    sw = SummaryWriter(
        save_tensorboard_dir)  # tensorboard --logdir='./tensorboard_server/s_1_adam0.0005'

    global_step = 0
    for epoch in range(cfg.epoch):
        net.train(True)

        current_step = 0
        max_iter = cfg.epoch * len(loader)
        for step, (image, mask) in enumerate(loader):

            image, mask = image.cuda().float(), mask.cuda().float()
            # with autocast():

            pred, PRM1_out, PRM2_out, PRM3_out, PRM4_out = net(image)

            loss_pred = loss(pred, mask)
            loss_PRM1_out = loss(PRM1_out, mask)
            loss_PRM2_out = loss(PRM2_out, mask)
            loss_PRM3_out = loss(PRM3_out, mask)
            loss_PRM4_out = loss(PRM4_out, mask)

            loss = loss_pred + loss_PRM1_out / 2 + loss_PRM2_out / 4 + loss_PRM3_out / 8 + loss_PRM4_out / 16

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            current_step += 1

            current_iter = epoch * len(loader) + step + 1
            current_lr = poly_learning_rate(cfg.lr, current_iter, max_iter, power=0.9)
            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr

            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'total_loss': loss.item(), 'loss_pred': loss_pred.item(),
                                    'loss_PRM1_out': loss_PRM1_out.item(),
                                    'loss_PRM2_out': loss_PRM2_out.item(),
                                    'loss_PRM3_out': loss_PRM3_out.item(),
                                    'loss_PRM4_out': loss_PRM4_out.item()}, global_step=global_step)

            if step % 10 == 0:
                print('%s | step:%d | current_step:%d | epoch:%d/%d | lr=%.6f | loss=%.6f' % (
                    datetime.datetime.now(), global_step, current_step, epoch + 1, cfg.epoch,
                    optimizer.param_groups[0]['lr'],
                    loss.item()))
            del loss, loss_pred, loss_PRM1_out, loss_PRM2_out, loss_PRM3_out, loss_PRM4_out, pred, PRM1_out, PRM2_out, PRM3_out, PRM4_out

        if epoch >= 10:
            torch.save(net.state_dict(), save_pth_dir + str(epoch + 1) + '.pth')

