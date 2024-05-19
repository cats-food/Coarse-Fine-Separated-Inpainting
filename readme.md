# Two-Stage Coarse-Fine-Separated Image Inpainting Network with Gated Attention (嵌入门控注意力的二阶段粗精分离图像修复模型 )



## Prerequisites
- Python 3.6
- PyTorch 1.2
- NVIDIA GPU + CUDA cuDNN
- Some other frequently-used python packages, like opencv, numpy, imageio, etc.

## Datasets prepration
We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris Street View](https://github.com/pathak22/context-encoder) datasets. 
The irregular mask dataset is available from [here](http://masc.cs.gmu.edu/wiki/partialconv).
After Downloading images and masks, create .filst file containing the dataset path in `./datasets/` (some examples have been given, refer to so).


## Pretraiend model
Download pretrained models from my [OneDrive](https://tjueducn-my.sharepoint.com/:f:/g/personal/yangshiyuan_tju_edu_cn/EklZ3296w8RIkThj48kU338Bged2g8KnjBxqiciPew0beA?e=FLUJNm),  and place .pth files in the `./checkpoints` directory.

## Training
Please edit the config file `./config.yml` for your training setting.
The options are all included in this file, see comments for the explanations. 

Once you've set up, run the `./train.py` script to launch the training.
```shell script
python train.py
```
## Testing
Download pretrained models as said above, use `./test.py` for testing:

```shell script
python test.py
--G1 <path to generator 1>
--G2 <path to generator 2>
--input <path to input images>
--mask <path to masks>
--output <path to results directory>
```



Alternatively, you can also edit these options in the config file `./config.yml`.
