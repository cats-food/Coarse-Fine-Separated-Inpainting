import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc import imread, imresize
#import cv2
from skimage.color import rgb2gray, gray2rgb
from skimage import io
from .utils import create_mask, tshow


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.input0_size = config.INPUT0_SIZE
        #self.sigma = config.SIGMA
        #self.edge = config.EDGE
        self.mask = config.MASK
        #self.nms = config.NMS
        self.model = config.MODEL
        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if not self.training:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size
        size0 = self.input0_size

        # load image
        img = io.imread(self.data[index]) # for psv and celeba


        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)  # 默認 bilinear
        #if self.model!=2
        img0 = scipy.misc.imresize(img, [size0, size0], interp='bilinear')    # add low-resolution img
        #img0 = cv2.resize(img, dsize=(size0,size0), interpolation=cv2.INTER_LINEAR)


        # create grayscale image
        #img_gray = rgb2gray(img)

        # load mask
        mask = self.load_mask(img, index)
        mask0 = scipy.misc.imresize(mask, [size0, size0], interp='nearest')
        #mask0 = cv2.resize(mask, (size0, size0), interpolation=cv2.INTER_LINEAR)


        #return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)
        return self.to_tensor(img), self.to_tensor(img0), self.to_tensor(mask), self.to_tensor(mask0)


    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = io.imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 128).astype(np.uint8) * 255       # threshold due to interpolation
            #ask = imresize(mask, size=(imgh, imgw), interp='nearest')
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = io.imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 128).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = scipy.misc.imresize(img, [height, width])
        #img = cv2.resize(img, dsize=(height, width))

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=False
            )

            for item in sample_loader:
                yield item
