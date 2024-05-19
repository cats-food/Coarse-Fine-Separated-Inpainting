import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import CoarseModel, RefineModel
from .utils import Progbar, create_dir, stitch_images, imsave, tshow
from .metrics import PSNR
from torchvision.utils import save_image as tsave


class Engine():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'inpaint1'
        elif config.MODEL == 2:
            model_name = 'inpaint2'
        elif config.MODEL == 3:
            model_name = 'inpaint1-2'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.debug = False
        self.model_name = model_name

        # self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.inpaint_model1 = CoarseModel(config).to(config.DEVICE)
        self.inpaint_model2 = RefineModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        #self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST,  config.TEST_MASK_FLIST, augment=False, training=False)
        else:  # train or val mode
            self.train_dataset = Dataset(config, config.TRAIN_FLIST,  config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST,  config.VAL_MASK_FLIST, augment=False, training=False)
            # self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        #self.samples_path = './samples'        # self.samples_path = os.path.join(config.PATH, 'samples')
        if os.path.exists(config.SAMPLE_PATH):
            self.samples_path = config.SAMPLE_PATH
        else:
            print('ysy warning: config.SAMPLE_PATH is invalid! samples will be saved to default path: ./samples')
            self.samples_path = './samples'

        #if config.RESULTS is not None:
        if os.path.exists(config.RESULTS):
            self.results_path = os.path.join(config.RESULTS)
        else:
            print('ysy warning: config.RESULTS is invalid! samples will be saved to default path: ./results')
            self.results_path = './results'  # self.results_path = os.path.join(config.PATH, 'results')

        if os.path.exists(config.LOG_PATH):
            self.log_train = os.path.join(config.LOG_PATH, 'log_train_' + model_name + '.dat')
            self.log_val = os.path.join(config.LOG_PATH, 'log_val_' + model_name + '.dat')
        else:
            print('ysy warning: LOG_PATH is invalid! samples will be saved to default path: ./')
            self.log_train = 'log_train_' + model_name + '.dat'
            self.log_val = 'log_val_' + model_name + '.dat'


        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        #self.log_file_val = os.path.join(config.LOG_PATH, 'log_' + model_name + '_val.dat')

    def load(self):
        if self.config.MODEL == 1:
            self.inpaint_model1.load()
            #self.edge_model.load()

        elif self.config.MODEL == 2:
            self.inpaint_model2.load()

        else:  # Model == 3 或 4， edge-inp 和 joint
            #self.edge_model.load()   # edge_model 包含边缘生成器和判别器
            self.inpaint_model1.load()
            self.inpaint_model2.load()  # inpaint_model2 也包含边缘生成器和判别器

    def save(self):
        if self.config.MODEL == 1:
            self.inpaint_model1.save()

        elif self.config.MODEL == 2 or self.config.MODEL == 3:
            self.inpaint_model2.save()

        else:
            #self.edge_model.save()
            self.inpaint_model1.save()
            self.inpaint_model2.save()

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            drop_last=True,
            shuffle=True
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            for items in train_loader: # 每个item都是一个包含4个张量的列表，items[0]残缺彩图； items[1]残缺灰度图  ；items[2]残缺边缘图  ；items[3] 二值mask
                #self.edge_model.train() # 这里的train方法是pytorch内置的，我估计是把网络调整成train的状态，这里并不是什么前向传播
                self.inpaint_model1.train()
                self.inpaint_model2.train()

                # images2, images_gray, edges, masks2 = self.cuda(*items) # 把items的4项放到gpu里，分别得到 images2（完整）, images_gray（完整）, edges（完整）, masks2
                images2, images1, masks2, masks1 = self.cuda(*items)
                                                                      # 注意，被mask弄残的images和edges是在edge_model或inpaintModel里面进行的（查看 models.py）


                # 此模式下inpaintModel1输入的。只更新inpaintModel1的G######################################################
                if model == 1:
                    # train
                    #outputs2, gen_loss, dis_loss, logs = self.edge_model.process(images_gray, edges, masks2)  # 前向传播，返回补全edge和损失，以及记录信息的log
                                                                                                            # 注意 此处的image和edge暂时是完整的，它俩稍后在edge_model里会被mask弄残（查看 models.py）
                    outputs1, gen_loss, dis_loss, logs = self.inpaint_model1.process(images1, masks1)
                    outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                    # metrics
                    psnr1 = self.psnr(self.postprocess(images1), self.postprocess(outputs1_merged))
                    mae1 = (torch.sum(torch.abs(images1 - outputs1_merged)) / torch.sum(images1)).float()
                    logs.append(('psnr1', psnr1.item()))
                    logs.append(('mae1', mae1.item()))
                    # backward
                    self.inpaint_model1.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model1.iteration


                #   此模式下inpaintModel2输入的是真值图目标区域部分降采样后又升采样的。只更新inpaintModel2的G和D ######################################################
                elif model == 2:
                    # train
                    #outputs2, gen_loss, dis_loss, logs = self.inpaint_model2.process(images2, edges, masks2) # 注意 此处的image和edge是完整的，其中image稍后在inpaintModel会被mask弄残（查看 models.py）
                    outputs2, gen_loss, dis_loss, logs = self.inpaint_model2.process(images2, images1,  masks2) # inp2 输入为：高分辨真值，低分辨图，高分辨mask。低分辨图将在forward里上采样融合
                    outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))

                    # metrics
                    psnr2 = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                    mae2 = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()
                    logs.append(('psnr2', psnr2.item()))
                    logs.append(('mae2', mae2.item()))

                    # backward
                    self.inpaint_model2.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model2.iteration


                # inpaint with edge model  此模式下edgeModel输入的images是残的，edge也是残的，补全edge后给inpaintModel用。inpaintModel输入的images是残的。
                # 该模式只更新inpaintModel的G和D，不会更新edgeModel的G和D     ######################################################
                elif model == 3:
                    with torch.no_grad():
                        outputs1 = self.inpaint_model1(images1,masks1)
                        outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                    outputs1_merged.detach_()
                    #with autocast():
                    outputs2, gen_loss, dis_loss, logs = self.inpaint_model2.process(images2, outputs1_merged, masks2)  # edgemodel输出的outputs的自动梯度被切断
                    # inp2 输入为：高分辨真值，低分辨图，高分辨mask。低分辨图将在forward里上采样融合
                    outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))

                    # metrics
                    psnr2 = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                    mae2 = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()
                    logs.append(('psnr2', psnr2.item()))
                    logs.append(('mae2', mae2.item()))

                    # backward
                    self.inpaint_model2.backward(gen_loss, dis_loss)
                    iteration = self.inpaint_model2.iteration   # 只更新inpaintModel，不更新edgeModel


                # joint model 此模式下edgeModel先补全边缘然后给inpModel。 会更新inpaintModel的G和D以及edgeModel的G和D
                elif model == 4:
                    # train
                    outputs1, gen1_loss, dis1_loss, logs1 = self.inpaint_model1.process(images1,masks1) # 注意 此处输入的image和edge都是完整的，被mask弄残的images和edges是在edge_model或inpaintModel里面完成的（查看 models.py）
                    outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                    outputs2, gen2_loss, dis2_loss, logs2 = self.inpaint_model2.process(images2, outputs1_merged, masks2)
                    outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))

                    # metrics
                    psnr1 = self.psnr(self.postprocess(images1), self.postprocess(outputs1_merged))
                    mae1 = (torch.sum(torch.abs(images1 - outputs1_merged)) / torch.sum(images1)).float()
                    psnr2 = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                    mae2 = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()

                    logs1.append(('psnr1', psnr1.item()))
                    logs1.append(('mae1', mae1.item()))
                    logs2.append(('psnr2', psnr2.item()))
                    logs2.append(('mae2', mae2.item()))
                    logs = logs1 + logs2

                    # backward
                    self.inpaint_model2.backward(gen2_loss, dis2_loss)
                    self.inpaint_model1.backward(gen1_loss, dis1_loss)
                    iteration = self.inpaint_model2.iteration

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images2), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval(iteration=iteration)


                if iteration >= max_iteration:
                    keep_training = False
                    break


        print('\nEnd training....')

    @ torch.no_grad()
    def eval(self, iteration=None):

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1, #self.config.BATCH_SIZE,  # 防止爆显存
            num_workers=0,
            drop_last=False,
            shuffle=False
        )

        model = self.config.MODEL
        # total = len(self.val_dataset)

        self.inpaint_model1.eval()
        self.inpaint_model2.eval()

        psnr_acc = 0
        mae_acc = 0

        # progbar = Progbar(total, width=20, stateful_metrics=['it'])
        idx = 0

        for items in val_loader:
            idx += 1
            # images2, images_gray, edges, masks2 = self.cuda(*items)
            images2, images1, masks2, masks1 = self.cuda(*items)

            if model == 1:
                with torch.no_grad():
                    outputs1 = self.inpaint_model1(images1, masks1)
                    outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                    outputs_merged = outputs1_merged
                psnr = self.psnr(self.postprocess(images1), self.postprocess(outputs1_merged))
                mae = (torch.sum(torch.abs(images1 - outputs1_merged)) / torch.sum(images1)).float()
                tsave(torch.cat((images1[0], masks1[0].repeat(3,1,1), outputs_merged[0]), dim=2),
                      os.path.join(self.samples_path, self.model_name+'-it' + str(iteration).zfill(8) + '-' + str(idx).zfill(2) + '.jpg'),
                      normalize=True, scale_each=True, range=(0, 1))

            elif model == 2:
                raise Exception('ysy: pending completion')

            # inpaint with edge model / joint model
            else:
                with torch.no_grad():
                    outputs1 = self.inpaint_model1(images1, masks1)
                    outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                    outputs2 = self.inpaint_model2(images2, outputs1_merged, masks2)
                    outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))
                    outputs_merged = outputs2_merged
                psnr = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                mae = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()
                tsave(torch.cat((images2[0], masks2[0].repeat(3,1,1), outputs_merged[0]), dim=2),
                      os.path.join(self.samples_path, self.model_name +'-it' + str(iteration).zfill(8) + '-' + str(idx).zfill(2) + '.jpg'),
                      normalize=True, scale_each=True, range=(0, 1))


            print('evaluating ', ' psnr = ', format(psnr.item(), '.2f'),',  mae = ', format(mae.item(), '.4f'))
            psnr_acc = psnr_acc + psnr
            mae_acc = mae_acc + mae

        mean_psnr = round(psnr_acc.item() / idx, 2)
        mean_mae = round(mae_acc.item() / idx, 4)
        print('mean psnr = ' + str(mean_psnr) + 'mean rmae = ' + str(mean_mae))

        with open(self.log_val, 'a') as f:
            f.write('iter ' + str(iteration) + ': mean psnr = ' + str(mean_psnr) + ' , mean mae = ' + str(mean_mae) + '\n')

    @ torch.no_grad()
    def test(self, multi_pth_eval = False):
        # self.edge_model.eval()  # 此处的eval方法是pytorch内置的，模型在测试的时候都要调用此方法
        self.inpaint_model1.eval()
        self.inpaint_model2.eval()

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        test_info = []
        for items in test_loader:   # 每个item都是一个包含4个张量的列表，items[0]残缺彩图； items[1]残缺灰度图  ；items[2]残缺边缘图  ；items[3] 二值mask
            name = self.test_dataset.load_name(index)  # name是当前item的文件名，如 0001.png
            #images2, images_gray, edges, masks2 = self.cuda(*items)  # 把items的4项从列表里拆解出来，放到gpu里，分别得到 images2（残）, images_gray（残）, edges（残）, masks2
            images2, images1, masks2, masks1 = self.cuda(*items)  # 把items的4项从列表里拆解出来，放到gpu里，分别得到 images2（残）, images_gray（残）, edges（残）, masks2
            #edges = 'ysy_占坑放报错用'

            index += 1

            if model == 1:
                outputs1 = self.inpaint_model1(images1, masks1)
                outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                outputs_merged = outputs1_merged
                psnr = self.psnr(self.postprocess(images1), self.postprocess(outputs1_merged))
                mae = (torch.sum(torch.abs(images1 - outputs1_merged)) / torch.sum(images1)).float()


            elif model == 2:

                outputs2 = self.inpaint_model2(images2, images1,  masks2)
                outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))
                outputs_merged = outputs2_merged
                psnr = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                mae = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()

            # inpaint with edge model / joint model
            else:
                outputs1 = self.inpaint_model1(images1, masks1).detach()
                outputs1_merged = (outputs1 * masks1) + (images1 * (1 - masks1))
                outputs2 = self.inpaint_model2(images2, outputs1_merged, masks2)
                outputs2_merged = (outputs2 * masks2) + (images2 * (1 - masks2))
                outputs_merged = outputs2_merged
                psnr = self.psnr(self.postprocess(images2), self.postprocess(outputs2_merged))
                mae = (torch.sum(torch.abs(images2 - outputs2_merged)) / torch.sum(images2)).float()

            output = self.postprocess(outputs_merged)[0]  # [0, 1] float B*C*H*W ======> [0, 255] int  B*H*W*C
            print(index, name, '   ', 'psnr= ', psnr.item(), '  mae= ', mae.item())
            #print(index, name)
            # save a copy of gt
            #imsave(self.postprocess(images2)[0],os.path.join(self.results_path, name[0:-4] + '_gt' + name[-4:]))
            # save masked image (gt*mask)
            #imsave(self.postprocess(images2 * (1 - masks2) + masks2)[0],os.path.join('E:\Desktop/batch_test_results\__MASKED__', name[0:-4] + '_masked' + name[-4:]))
            #imsave(self.postprocess(images2 * (1 - masks2) + masks2)[0],os.path.join(self.results_path, name[0:-4] + '_masked' + name[-4:]))
            # save inpainted result
            if not multi_pth_eval:# 默认为False normal test
                imsave(output, os.path.join(self.results_path, name[0:-4]+'_out'+name[-4:]))
            else: # fpr multi-pth eval
                if model == 1:
                    iter1 = str(self.inpaint_model1.iteration)
                    iter2 = 'nan'
                    tmp_iter = iter1
                elif model == 2:
                    iter1 = 'nan'
                    iter2 = str(self.inpaint_model2.iteration)
                    tmp_iter = iter2
                else:
                    iter1 = str(self.inpaint_model1.iteration)
                    iter2 = str(self.inpaint_model2.iteration)
                    tmp_iter = iter2
                imsave(output, os.path.join(self.results_path, name[0:-4] + '-' +tmp_iter.zfill(8) + name[-4:]))
                test_info.append(['# '+iter1,'# '+iter2,index,name,psnr.item(),mae.item()])  # 测试样本数hang的二维列表  # 每一行分别是： G1-iter   G2-iter  idx    name   psnr    mae

            # if self.debug:
            #     edges = self.postprocess(1 - edges)[0]
            #     masked = self.postprocess(images2 * (1 - masks2) + masks2)[0]
            #     fname, fext = name.split('.')
            #
            #     imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
            #     imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))
        print('\ntest results are saved to: %s' %(self.results_path))
        print('\nEnd test....')
        return test_info

    def log(self, logs):
        with open(self.log_train, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
