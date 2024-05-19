import os
import argparse
import numpy as np

# 该脚本读取数据集路径src里第begin个到第end（不含）个样本，（不指定则读取全部），然后生成一个包含数据集路径的flist文件（类似txt）

# 常用路径
#     python scripts/flist.py  --src E:\DataSet_CelebA\256x256-png   --dst ./datasets/celeba_train.flist
#     python scripts/flist.py  --src E:\DataSet_ParisStreetView\paris_train_256_cropped   --dst ./datasets/psv_train.flist

#     python scripts/flist.py  --src E:\DataSet_nvMask\mask-256-test  --dst ./datasets/masks_train.flist



parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, help='source path to the dataset')
parser.add_argument('--dst', type=str, help='destination path to the file list')
parser.add_argument('--begin', type=int, default=0, help='从第begin个样本开始取, 默认从0开始读')  # ysy 加入
parser.add_argument('--end', type=int, default=None, help='取到第end个样本（不含），默认读到最后一个')  # ysy 加入
args = parser.parse_args()

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

begin = args.begin
end =args.end

images = []

for root, dirs, files in os.walk(args.src):
    print('loading ' + root)
    if end is None or end>len(files):
        end = len(files)

    for file in files[begin:end]:
        if os.path.splitext(file)[1].upper() in ext:
            images.append(os.path.join(root, file))



images = sorted(images)
np.savetxt(args.dst, images, fmt='%s')
print('done')