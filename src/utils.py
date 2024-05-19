import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
from PIL import Image


#自己写的tensor图片可视化函数，调试用
def tshow(*imgs):
    '''
     输入可以是张量，numpy数组，本函数可以将它们show出来
     示例：
     tshow(a) # a可以是torch.Tensor也可以是numpy.ndarray
        当a是torch.Tensor时，如果是四维，按bz维分成bz个subplot，每个subplot最多显示三个通道（RGB格式）；如果是3维张量，通道维数最多显示三个通道（RGB格式）；如果是2维张量，按灰度图显示
        当a是numpy.ndarray时，如果三维，通道数可以为1或3，以RGB显示；如果是二维，则以灰度显示。
     tshow(a,b,c) 可以生成三个窗口，分别显示a,b,c。三者可以是tensor或ndarray的混合
    '''
    img_idx = 0
    for img in imgs:
        img_idx +=1
        plt.figure(img_idx)
        if isinstance(img, torch.Tensor):  # 判断是否是torch张量
            img = img.detach().cpu()

            if img.dim()==4:
                bz = img.shape[0]
                c = img.shape[1]
                if bz==1 and c==1:  #4维张量，单张灰度图
                    img=img.squeeze()
                elif bz==1 and c==3: #4维张量，单张彩图
                    img=img.squeeze()
                    img=img.permute(1,2,0)
                elif bz==1 and c > 3: # 4维张量，多张 【特征图】
                    img = img[:,0:3,:,:]
                    img = img.permute(0, 2, 3, 1)[:]
                    print('ysy warning: more than 3 channels! only channels 0,1,2 are preserved!')
                elif bz > 1 and c == 1:  # 4维张量，多张灰度图
                    img=img.squeeze()
                elif bz > 1 and c == 3:  # 4维张量，多张彩图
                    img = img.permute(0, 2, 3, 1)
                elif bz > 1 and c > 3:  # 4维张量，多张 【特征图】
                    img = img[:,0:3,:,:]
                    img = img.permute(0, 2, 3, 1)[:]
                    print('ysy warning: more than 3 channels! only channels 0,1,2 are preserved!')
                else:
                    raise Exception("ysy: Invalid type!  " + str(img.size()))
            elif img.dim()==3:
                bz = 1
                c = img.shape[0]
                if c == 1:  # 3维张量，单张灰度图
                    img=img.squeeze()
                elif c == 3:  # 3维张量，单张彩图
                    img = img.permute(1, 2, 0)
                else:
                    raise Exception("ysy: Invalid type!  " + str(img.size()))
            else:
                raise Exception("ysy: Invalid type!  "+str(img.size()))

            img = img.numpy()  # 转化成numpy
            img = img.squeeze()
            if bz ==1:
                plt.imshow(img, cmap='gray')
                # plt.colorbar()
                # plt.show()
            else:
                for idx in range(0,bz):
                    plt.subplot(int(bz**0.5),int(np.ceil(bz/int(bz**0.5))),int(idx+1))
                    plt.imshow(img[idx], cmap='gray')


        elif isinstance(img, np.ndarray): # 如果是numpy, PIL, opencv读取的是numpy.ndarray格式
            img = img.squeeze()
            plt.imshow(img, cmap='gray')
        else:
            raise Exception("ysy: Invalid type:  "+str(type(img)))
    plt.show()





#自己写的直接把图片读成tensor格式的函数
def ysyread(img):
    raise Exception('ysy: Pending complete ...')


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    # These 2 lines was commented by ysy since they bring unexpected color spot on saved images.
    # im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    # im.save(path)

    # use np.clip to cutoff values up to 255, avoid overflow.
    scipy.misc.imsave(path,np.clip(img.cpu().numpy().squeeze(),0,255).astype(np.uint8))


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


def check_path_existence(args): # bad implementation, needs to be improved
    for k,v in args.items():
        if isinstance(v, str):
            if '/' in v and '.' not in os.path.basename(v): # this is (probably) a directory
                os.makedirs(v, exist_ok=True)  # if it doesnt exist, create one



