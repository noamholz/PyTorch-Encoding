###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import sys
import numpy as np
import pandas as pd
import random
import math
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform
from encoding.models.sseg.base import BaseNet
from encoding.datasets.base import BaseDataset

from data_prep.conf_labels import labels as rgo_annots_labels
from train_predict.dataloader import mask_reassign_labels
from train_predict import preprocessing
from functools import partial
from train_predict import undistort_fisheye
labels_conversion_dict = {'else': 0, 'background': 0,
                          'grass_non': 0, 'grass_non_local': 0,
                          'too_dark_to_say': -1,
                          'grass_unknown': -1,
                          'grass_synth': -1, 'grass_sparse': -1,
                          'pavement_line': -1,
                          'grass_border_out': 1, 'grass_border_in': 1, 'grass': 1,
                          'sky_border_out': 0, 'sky_border_in': 0, 'sky': 0
                          }



class RGoGrassSegmentation(BaseDataset):
    BASE_DIR = '?'
    NUM_CLASS = 2  #150  #2
    def __init__(self, root=os.path.expanduser(
        '../../../../../../../datasets/annots_findgrass/rgo_annots_20210121_color'), split='train',
                 mode=None, transform=None, target_transform=None,
                 df_path='../../../../results/20210126-153039_PSPNet101_bs2_noAccum_20210121data_713x713/df.csv', **kwargs):
        super(RGoGrassSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        self.aux_label = -1
        self.base_shape = (kwargs['base_size'], kwargs['base_size'], 3)
        self.crop_shape = (kwargs['crop_size'], kwargs['crop_size'], 3)
        self.out_shape = (kwargs['base_size'], kwargs['base_size'], 3)
        self.dark_satur_threshold = {'low': 30, 'high': 999}
        self.metrbox = [330, 0, 470, 800]
        self.augs_dict = {'train':
                         [partial(preprocessing.crop_resize_to_standard,
                                  target_shapes={'img': self.crop_shape, 'msk': tuple(self.crop_shape[:-1]) + (1,)},
                                  method='no_crop', aux_label=self.aux_label),
                          # partial(undistort_fisheye.Fisheye(self.crop_shape[:-1], padval=200, w=0.92, f=300).randRot,
                          #         aux_label=self.aux_label, p=0.5),
                          # partial(preprocessing.crop_img_msk,
                          #         crop_pxls={'y': [self.im_shape_orig[0] - self.im_shape_cropped[0], None],
                          #                    'x': [0, self.crop_shape[1]]}),
                          # partial(preprocessing.sky_rep, df_skies_samps=skies_synth['df_skies'], extra_sky=skies_synth['extra_sky'], p=1),
                          # partial(preprocessing.rand_shadows, kernel_size=291, sigma=50, p=0.5),
                          partial(preprocessing.apply_mark_dark_satur_regions,
                                  cond_msk_val=labels_conversion_dict['grass'],
                                  threshs=self.dark_satur_threshold, rot_img=False, min_size=100, fill_value=2),
                          partial(preprocessing.rand_shadows2,
                                  minIthresh=self.dark_satur_threshold,
                                  projective_matrix=preprocessing.prep_perspective_projmat(self.crop_shape),
                                  xy_meshes=np.stack(
                                      np.meshgrid(range(self.crop_shape[1]), range(self.crop_shape[0]))),
                                  thresh=20, blur=(3, 30), p=0.5),
                          partial(preprocessing.random_saturation_scaling, fact=0.1, p=0.5),
                          partial(preprocessing.rand_dummy_blobs, percentile=20, aux_label=self.aux_label, p=0.5),
                          partial(preprocessing.gauss_noise, std=5, p=0.3),
                          partial(preprocessing.album_wraper, preprocessing.aug_med, p=1),
                          partial(preprocessing.apply_mark_dark_satur_regions,
                                  cond_msk_val=labels_conversion_dict['grass'],  # labels_conversion_dict['grass'],
                                  threshs=self.dark_satur_threshold, rot_img=False, min_size=100, fill_value=2),
                          partial(preprocessing.random_hist_scale,
                                  ROIbox=(np.array(self.metrbox) * np.concatenate(
                                      [(1,1)] * 2)).astype(int)),
                          partial(preprocessing.ground_gaussian_spot, coords='horizon', shape=2, sigma=[50, 150], p=0),
                          partial(preprocessing.fixed_perspective, crop=False, p=0),
                          partial(preprocessing.resize_img_msk, target_shapes={'msk': self.out_shape}),
                          ],
                     'valid':
                         [partial(preprocessing.crop_resize_to_standard,
                                  target_shapes={'img': self.crop_shape, 'msk': tuple(self.crop_shape[:-1]) + (1,)},
                                  method='no_crop', aux_label=self.aux_label),
                          # partial(preprocessing.crop_img_msk,
                          #         crop_pxls={'y': [kwargs['base_size'][0] - self.im_shape_cropped[0], None], 'x': [0, self.crop_shape[1]]}),
                          partial(preprocessing.fixed_perspective, crop=False, p=0.0),
                          partial(preprocessing.resize_img_msk, target_shapes={'msk': self.out_shape}),
                          ],
                     'test':
                         [partial(preprocessing.crop_img_msk,
                                  crop_pxls={'y': [self.base_shape[0] - self.crop_shape[0], None],
                                             'x': [0, self.crop_shape[1]]}),
                          partial(preprocessing.resize_img_msk, target_shapes={'msk': self.out_shape}),
                          ],
                     'invert_preproc':
                         [
                             partial(preprocessing.resize_img_msk,
                                     target_shapes={'img': self.base_shape, 'msk': self.base_shape[:-1] + (1,)}),
                             # partial(preprocessing.crop_image_undo,
                             #         crop_pxls={'y': [kwargs['base_size'][0] - self.im_shape_cropped[0], None], 'x': [0, self.crop_shape[1]]},
                             #         orig_shape=kwargs['base_size']),
                         ]
                     }

        self.datasets_path = root[:root.find('datasets/') + 8]
        if split == 'val':
            split = 'valid'
        if not df_path:
            # assert exists and prepare dataset automatically
            root = os.path.join(root, self.BASE_DIR)
            assert os.path.exists(root), "Please setup the dataset using" + \
                "encoding/scripts/prepare_rgo.py"
            self.images, self.masks = _get_rgo_pairs(root, split)
            self.df = None
        else:
            self.df = pd.read_csv(df_path)  #.iloc[:50]
            df_split = self.df.loc[self.df.datasettype == split, :]
            typ_img_path = df_split.img.values[0]
            df_split.img = df_split.img.str.replace(typ_img_path[:typ_img_path.find('datasets/') + 8], self.datasets_path)
            df_split.msk = df_split.msk.str.replace(typ_img_path[:typ_img_path.find('datasets/') + 8], self.datasets_path)
            print(typ_img_path[:typ_img_path.find('datasets/') + 8], self.datasets_path, df_split.img.values[0])
            self.images, self.masks = df_split.img.values, df_split.msk.values
        if split != 'test':
            assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        assert img is not None, "error in reading image: " + self.images[index]
        assert mask is not None, "error in reading mask: " + self.masks[index]
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(2, 1)
        # # print(np.array(img).shape)
        # ax[0].imshow(np.array(img))
        # ax[1].imshow(np.array(mask))
        # plt.show()
        # synchrosized transform

        if self.mode == 'train':
            # img, mask = self._sync_transform2(img, mask)
            img, mask = np.array(img), np.array(mask)[..., None]
            for f_aug in self.augs_dict['train']:
                img, mask = f_aug(img, mask)
            img, mask = Image.fromarray(img.astype(np.uint8)), Image.fromarray(mask.astype(np.uint8)[..., 0])
            mask = self._mask_transform(mask)
        elif self.mode in ['val', 'valid']:
            for f_aug in self.augs_dict['valid']:
                img, mask = f_aug(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask, [self.images[index], self.masks[index]]

    # import matplotlib.pyplot as plt
    # img2 = img.cpu().numpy().copy() * np.array([0.229, 0.224, 0.225])[..., None, None] + \
    #        np.array([0.485, 0.456, 0.406])[..., None, None]
    # plt.imshow(np.moveaxis(img2, 0, -1))
    # plt.show()

    def _sync_transform2(self, img, mask):
       # random mirror
       if random.random() < 0.5:
           img = img.transpose(Image.FLIP_LEFT_RIGHT)
           mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
       crop_size = self.crop_size
       # random scale (short edge)
       w, h = img.size
       long_size = random.randint(int(self.base_size*0.9), int(self.base_size*1.1))
       if h > w:
           oh = long_size
           ow = int(1.0 * w * long_size / h + 0.5)
           short_size = ow
       else:
           ow = long_size
           oh = int(1.0 * h * long_size / w + 0.5)
           short_size = oh
       img = img.resize((ow, oh), Image.BILINEAR)
       mask = mask.resize((ow, oh), Image.NEAREST)
       # pad crop
       if short_size < crop_size:
           padh = crop_size - oh if oh < crop_size else 0
           padw = crop_size - ow if ow < crop_size else 0
           img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
           mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
       # random crop crop_size
       w, h = img.size
       x1 = random.randint(0, w - crop_size)
       y1 = random.randint(0, h - crop_size)
       img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
       mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
       # gaussian blur as in PSP
       if random.random() < 0.5:
           img = img.filter(ImageFilter.GaussianBlur(
               radius=random.random()))
       # final transform
       return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        target = mask_reassign_labels(np.array(mask, dtype=np.int64), rgo_annots_labels, labels_conversion_dict)
        # mask = Image.fromarray(mask)
        # target = np.array(mask).astype('int64') # - 1
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1


def _get_rgo_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".png"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print('len(img_paths):', len(img_paths))
        # assert len(img_paths) == 20210
    elif split in ['val', 'valid']:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        # assert len(img_paths) == 2000
    else:
        assert split == 'trainval'
        train_img_folder = os.path.join(folder, 'images/training')
        train_mask_folder = os.path.join(folder, 'annotations/training')
        val_img_folder = os.path.join(folder, 'images/validation')
        val_mask_folder = os.path.join(folder, 'annotations/validation')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
        assert len(img_paths) == 22210
    return img_paths, mask_paths
