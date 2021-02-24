###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import copy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import cv2
# from ....encoding import utils
# from ....encoding.nn import SegmentationLosses, SyncBatchNorm
# from ....encoding.parallel import DataParallelModel, DataParallelCriterion
# from ....encoding.datasets import get_dataset
# from ....encoding.models import get_segmentation_model

# import importlib.util
# import sys
# # # For illustrative purposes.
# # import tokenize
# # file_path = tokenize.__file__  # returns "/path/to/tokenize.py"
# # module_name = tokenize.__name__  # returns "tokenize"
#
# spec = importlib.util.spec_from_file_location('__init__', 'encoding/models/__init__.py')
# module = importlib.util.module_from_spec(spec)
# sys.modules['__init__'] = module
# spec.loader.exec_module(module)

import external_repos.pytorch_encoding.encoding.utils as utils
from external_repos.pytorch_encoding.encoding.nn import SegmentationLosses, SyncBatchNorm
from external_repos.pytorch_encoding.encoding.parallel import DataParallelModel, DataParallelCriterion
from external_repos.pytorch_encoding.encoding.datasets import get_dataset
from external_repos.pytorch_encoding.encoding.models import get_segmentation_model
# from rgo_grass import RGoGrassSegmentation
# from encoding.models.sseg.deeplab import DeepLabV3
from deeplab import get_deeplab

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='ade20k',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=520,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=480,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--se-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=16,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        parser.add_argument('--pretrained', action='store_true', default= False,
                            help='load pretrained weights')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument('--test-val', action='store_true', default= False,
                            help='generate masks on val set')
        parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'coco': 30,
                'pascal_aug': 80,
                'pascal_voc': 50,
                'pcontext': 80,
                'ade20k': 180,
                'citys': 240,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.lr is None:
            lrs = {
                'coco': 0.004,
                'pascal_aug': 0.001,
                'pascal_voc': 0.0001,
                'pcontext': 0.001,
                'ade20k': 0.004,
                'citys': 0.004,
                # 'rgo_grass': 0.006,
            }
            args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size
        print(args)
        return args


class Trainer():
    def __init__(self, args):
        self.args = args
        self.metric_dict = dict()
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.inverse_transform = lambda img: img * np.array([0.229, 0.224, 0.225])[..., None, None] + \
                           np.array([0.485, 0.456, 0.406])[..., None, None]
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        # if args.dataset != 'rgo_grass':
        trainset = get_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        testset = get_dataset(args.dataset, split='val', mode ='val', **data_kwargs)
        # else:
        #     trainset = RGoGrassSegmentation(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        #     testset = RGoGrassSegmentation(args.dataset, split='valid', mode='train', **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}

        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, worker_init_fn=self.worker_init_fn,
                                           **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, worker_init_fn=self.worker_init_fn,
                                         **kwargs)
        self.nclass = trainset.num_class
        self.df_res = pd.DataFrame(columns=['epoch', 'acc'])
        # model
        # if args.dataset != 'rgo_grass':
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone = args.backbone, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = SyncBatchNorm,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       pretrained=args.load_pretrained,
                                       root='../../pretrained_models')
        # else:
        #     model = get_deeplab(dataset=args.dataset,
        #                                    backbone = args.backbone, aux = args.aux,
        #                                    se_loss = args.se_loss, norm_layer = SyncBatchNorm,
        #                                    base_size=args.base_size, crop_size=args.crop_size)
        # print(model)
        if os.path.isdir('runs/rgo_grass/deeplab/resnest50/default'):
            for nn in range(100):
                if not os.path.isdir('runs/rgo_grass/deeplab/resnest50/default_bck{}'.format(nn)):
                    os.rename('runs/rgo_grass/deeplab/resnest50/default', 'runs/rgo_grass/deeplab/resnest50/default_bck{}'.format(nn))
                    break
        utils.save_scripts(args)
        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        optimizer = torch.optim.SGD(params_list, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
        # criterions
        self.criterion = SegmentationLosses(se_loss=args.se_loss,
                                            aux=args.aux,
                                            nclass=self.nclass, 
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight)
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        # resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.scheduler = utils.LR_Scheduler_Head(args.lr_scheduler, args.lr,
                                                 args.epochs, len(self.trainloader))
        self.best_pred = 0.0

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
    # def viz_result(self, img, prd, id_in_batch):
    #     fig, ax = plt.subplots(2, 1)
    #     img2 = trainer.inverse_transform(img[id_in_batch, ...].cpu().numpy().copy())
    #     img2 = np.moveaxis(img2, 0, -1)
    #     out2 = prd[id_in_batch, 1, ...].cpu().detach().numpy().copy()
    #     ax[0].imshow(img2)
    #     ax[1].imshow(out2)
    #     plt.show()
    #     return img2, out2

    def results2export(self, img, prd, tgt, id_in_batch=1):
        img2 = 255 * trainer.inverse_transform(img[id_in_batch, ...].cpu().numpy().copy())
        # print(img.min(), img.max(), ' --- ', img2.min(), img2.max())
        img2 = np.moveaxis(img2, 0, -1).astype(np.uint8)[..., ::-1]
        prd2 = torch.max(prd[id_in_batch, ...], 0)[1].cpu().numpy().astype(np.uint8)
        # prd2 = prd[id_in_batch, 1, ...].cpu().detach().numpy().copy()
        # print('8'*8)
        # print(prd2.shape, prd3.shape)
        tgt[tgt > 0] = 255
        tgt[tgt < 0] = 100
        tgt2 = tgt.numpy().astype(np.uint8)[id_in_batch, ...]

        return img2, prd2, tgt2


    def eval_batch(self, image, target):
        outputs = self.model(image)
        outputs = gather(outputs, 0, dim=0)
        # pred = outputs[0]
        target = target.cuda()
        correct, labeled = utils.batch_pix_accuracy_rgo(outputs.data, target, args.metric_box)
        inter, union = -1, -1
        # correct, labeled = utils.batch_pix_accuracy(pred.data, target)
        # inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
        return correct, labeled, inter, union

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        for i, (image, target, paths) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            with torch.no_grad():
                correct, labeled, inter, union = self.eval_batch(image, target)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            outputs = self.model(image)[0]
            loss = self.criterion(outputs, target)
            loss.backward()
            if (i % 50) == 0:
                pixAcc_i = 1.0 * correct / (np.spacing(1) + labeled)
                # try:
                img_out, prd_out, tgt_out = self.results2export(image, outputs, target, 0)
                # import matplotlib.pyplot as plt
                # plt.imshow(img_out)
                # plt.show()

                utils.save_images('trn_ep{}_i{}_img0.png'.format(epoch, i), img_out, args)
                utils.save_images('trn_ep{}_i{}_acc{:.2f}_prd0.png'.format(epoch, i, pixAcc_i), prd_out * 255, args)
                utils.save_images('trn_ep{}_i{}_tgt0.png'.format(epoch, i), tgt_out, args)
                # except Exception as ee:
                #     print(ee)
                #     print(paths[0], [im.shape for im in image], [p.shape for p in outputs], [t.shape for t in target])

            # print(np.isnan(loss.cpu().detach().numpy().item()), loss.item())
            # print(paths)
            if np.isnan(loss.cpu().detach().numpy().item()):
                print(paths)
                break
            # if loss.cpu().detach().numpy().item() == np.nan:
            #     print(paths)
            #     break
            self.optimizer.step()
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            train_loss += loss.item()

            tbar.set_description('Train loss: %.3f, pixAcc: %.3f, mIoU: %.3f' % (train_loss / (i + 1), pixAcc, mIoU))
            self.metric_dict.update({'epoch': epoch})
            self.metric_dict.update({'loss': train_loss / (i + 1), 'accuracy': pixAcc, 'mIoU': mIoU})
        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


    def validation(self, epoch):
        # Fast test during the training
        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        valid_loss = 0.0
        for i, (image, target, paths) in enumerate(tbar):
            with torch.no_grad():
                outputs = self.model(image)
                outputs = gather(outputs, 0, dim=0)
                loss = self.criterion(outputs, target)
                correct, labeled, inter, union = self.eval_batch(image, target)
                if i < 2:
                    pixAcc_i = 1.0 * correct / (np.spacing(1) + labeled)
                    img_out, prd_out, tgt_out = self.results2export(image, outputs, target, 0)
                    utils.save_images('val_ep{}_i{}_img0.png'.format(epoch, i), img_out, args)
                    utils.save_images('val_ep{}_i{}_acc{:.2f}_prd0.png'.format(epoch, i, pixAcc_i), prd_out * 255, args)
                    utils.save_images('val_ep{}_i{}_tgt0.png'.format(epoch, i), tgt_out * 255, args)


            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            valid_loss += loss.item()
            tbar.set_description(
                'Valid loss: %.3f, pixAcc: %.3f, mIoU: %.3f' % ((valid_loss / (i + 1), pixAcc, mIoU)))
            # rec_res.append(pixAcc)
        self.metric_dict.update({'val_loss': valid_loss / (i + 1), 'val_accuracy': pixAcc, 'val_mIOU': mIoU})


        new_pred = pixAcc  #(pixAcc + mIoU)/2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, self.args, is_best)


if __name__ == "__main__":
    args = Options().parse()
    # example run: 'python train.py --dataset ade20k --model encnet --aux --se-loss'
    args.dataset = 'rgo_grass'
    args.model = 'deeplab'
    args.backbone = 'resnest50'
    # args.ft = True
    args.resume = '../../experiments/segmentation/runs/rgo_grass/deeplab/resnest50/default_pretrained_binaryHead_noCrop/checkpoint.pth.tar'
    args.base_size = 800
    args.crop_size = 800
    args.batch_size = 2
    args.test_batch_size = 2
    args.epochs = 400
    args.lr = 0.005
    args.load_pretrained = True
    args.metric_box = [330, 0, 470, 800]
    #
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    if args.eval:
        trainer.validation(trainer.args.start_epoch)
    else:
        for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)
            trainer.df_res = trainer.df_res.append(pd.DataFrame(trainer.metric_dict, index=[0]))
            utils.save_log_csv(trainer.df_res, args)
