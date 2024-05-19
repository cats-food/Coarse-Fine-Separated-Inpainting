import os
from logging import exception

import cv2
import random
import numpy as np
import torch
import argparse
import src.utils as utils
from shutil import copyfile
from src.config import Config
from src.engine import Engine


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)

    # check path validity
    # assert os.path.exists(config.G_SAVE_PATH)
    # assert os.path.exists(config.D_SAVE_PATH)
    # assert os.path.exists(config.CoarseModel_G_LOAD_PATH)
    # assert os.path.exists(config.RefineModel_G_LOAD_PATH)
    utils.check_path_existence(config)


    print('you are using model: '+ str(config.MODEL))
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")



    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = Engine(config) # config文件指定加载哪些模型(4种)，运行哪种模式（train test eval），使用什么硬件(cpu gpu),数据集flist的路径等。传入类EdgeConnect中得到实例化的对象model
    model.load()  # 上一行是搭建模型骨架（还没有值），这一行是在模型上加载预训练模型的参数，至此model已经完全确定


    # model training
    if config.MODE == 1:
        # config.print()  # print config file
        print('\nstart training...\n')
        model.train() #因为model是类Edgeconnect的实例化对象，所以其train方法在edge_connect.py里实现

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test() #因为model是类Edgeconnect的实例化对象，所以其test方法在edge_connect.py里实现

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval() #因为model是类Edgeconnect的实例化对象，所以其eval方法在edge_connect.py里实现


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], help='1: inp1 model, 2: inp2 model, 3: inp1-2 model, 4: joint model')

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--G1', type=str, help='path to G1')
        parser.add_argument('--G2', type=str, help='path to G2')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()

    # load config file
    config_path = './config.yml' # 原config_path = os.path.join(args.path, 'config.yml')
    if not os.path.exists(config_path):
        raise Exception('ysy: config file not found!')

    config = Config(config_path)


    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        #config.MODEL = args.model if args.model is not None else 3
        # config.INPUT_SIZE = 0
        if args.model is not None:
            config.MODEL = args.model
        if args.G1 is not None:
            config.CoarseModel_G_LOAD_PATH = args.G1
        if args.G2 is not None:
            config.RefineModel_G_LOAD_PATH = args.G2
        if args.input is not None:
            config.TEST_FLIST = args.input
        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask
        if args.output is not None:
            config.RESULTS = args.output


    # eval mode
    elif mode == 3:
        #config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    main()
