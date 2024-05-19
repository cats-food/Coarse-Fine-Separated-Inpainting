import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .networks import CoarseGenerator, RefineGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, PerceptualStyleLoss
from .utils import tshow


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name  # name有两种值，'CoarseModel' 和 'RefineModel'。取决于是继承该BaseModel的是 CoarseModel类 还是RefineModel类（见下）
        self.config = config
        self.iteration = 0

        if name == 'CoarseModel':
            self.gen_weights_path = config.CoarseModel_G_LOAD_PATH
            if config.ENABLE_D1:
                self.dis_weights_path = config.CoarseModel_D_LOAD_PATH
        elif name == 'RefineModel':
            self.gen_weights_path = config.RefineModel_G_LOAD_PATH
            self.dis_weights_path = config.RefineModel_D_LOAD_PATH
        else:
            raise Exception('ysy: bug!')

    def load(self):  # 加载预训练模型的参数
        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator from path %s ...' % (self.name, self.gen_weights_path))

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)  # 此处data是个字典，只有俩键： 键'iteration'的值是迭代次数；键'generator'的值是生成器的参数
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'],
                                           strict=True)  # torch.load()只返回参数字典，还需要用模型实例的load_state_dict方法加载参数字典才行
            self.iteration = data['iteration']
        else:
            raise Exception('ysy: no model found at: ' + self.gen_weights_path)

        # load discriminator only when training
        if self.config.MODE == 1:
            try:
                if not os.path.exists(self.dis_weights_path):
                    raise Exception('ysy: no model found at: ' + self.dis_weights_path)

                print('Loading %s discriminator from path %s ...' % (self.name, self.dis_weights_path))
                if torch.cuda.is_available():
                    data = torch.load(self.dis_weights_path)
                else:
                    data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

                self.discriminator.load_state_dict(data['discriminator'])
            except AttributeError:
                print('ysy prompt: %s has no attribute \'dis_weights_path\' thus it will not be loaded' % self.name)

    def save(self):
        print('\nsaving %s...\n' % self.name)
        torch.save({
            'iteration': self.iteration,
            'generator': self.generator.state_dict()
        }, os.path.join(self.config.G_SAVE_PATH,
                        self.name + '_G_' + ('%08d' % self.iteration) + '.pth'))  # 原路径 self.gen_weights_path
        try:
            torch.save({
                'discriminator': self.discriminator.state_dict()
            }, os.path.join(self.config.D_SAVE_PATH,
                            self.name + '_D_' + ('%08d' % self.iteration) + '.pth'))  # 原路径 self.dis_weights_path
        except AttributeError:
            print('ysy prompt: %s has no attribute \'discriminator\' thus it will not be saved' % self.name)


class CoarseModel(BaseModel):
    def __init__(self, config):
        super(CoarseModel, self).__init__('CoarseModel', config)
        generator = CoarseGenerator()
        if config.ENABLE_D1:
            discriminator = Discriminator(in_channels=3, use_sigmoid= not (config.GAN_LOSS == 'hinge' or config.GAN_LOSS == 'wgan'))   # hinge 和 wgan 不用sigmoid
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            if config.ENABLE_D1:
                discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        # perceptual_loss = PerceptualLoss()
        # style_loss = StyleLoss()


        self.add_module('generator', generator)
        self.add_module('l1_loss', l1_loss)
        self.gen_optimizer = optim.Adam(params=generator.parameters(), lr=float(config.LR),betas=(config.BETA1, config.BETA2))

        if config.ENABLE_D1:
            self.add_module('discriminator', discriminator)
            self.adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
            self.dis_optimizer = optim.Adam(params=discriminator.parameters(), lr=float(config.LR) * float(config.D2G_LR), betas=(config.BETA1, config.BETA2))
        # self.add_module('perceptual_loss', perceptual_loss)
        # self.add_module('style_loss', style_loss)
        # self.add_module('adversarial_loss', adversarial_loss)

        # if self.config.GAN_LOSS == 'wgan':   # wgan使用rmsprop
        #     self.gen_optimizer=optim.RMSprop(params=generator.parameters(),lr=float(config.LR))#betas=(config.BETA1, config.BETA2))
        #     self.dis_optimizer=optim.RMSprop(params=discriminator.parameters(),lr=float(config.LR) * float(config.D2G_LR))#,???betas=(config.BETA1, config.BETA2))
        # else:    # 默认使用adam

    def process(self, images, masks):  # 在train和eval时调用，test时不会
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        # self.dis_optimizer.zero_grad()  # this line has been moved downward

        # process outputs
        outputs = self(images, masks)  # 喂给generator
        gen_loss = 0
        dis_loss = torch.tensor(float('nan'))
        gen_gan_loss = torch.tensor(float('nan'))

        if self.config.ENABLE_D1:
            self.dis_optimizer.zero_grad()
            dis_loss = 0
            # discriminator loss
            dis_input_real = images
            dis_input_fake = outputs.detach()
            dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
            dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
            dis_real_loss = self.adversarial_loss(dis_real, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

            # generator adversarial loss
            gen_input_fake = outputs
            gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
            gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_loss += gen_gan_loss

        # generator l1 loss
        # gen_l1_loss = self.l1_loss(outputs, images) / torch.mean(masks)  # original
        gen_l1_loss = 6 * torch.mean(torch.abs(outputs - images) * masks) + torch.mean(torch.abs(outputs - images) * (1 - masks))  # 6 hole_loss + 1 valid_loss
        gen_loss += gen_l1_loss * self.config.L1_LOSS_WEIGHT

        # # generator perceptual loss
        # gen_content_loss = self.perceptual_loss(outputs, images)
        # gen_loss += gen_content_loss* self.config.PERCEP_LOSS_WEIGHT
        #
        #
        # # generator style loss
        # gen_style_loss = self.style_loss(outputs * masks, images * masks)
        # gen_loss += gen_style_loss * self.config.STYLE_LOSS_WEIGHT

        # create logs
        logs = [
            ("l_d1", dis_loss.item()),
            ("l_g1", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            # ("l_per", gen_content_loss.item()),
            # ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs


    def forward(self, images, masks):  # train, eval, test 最后都会到这里
        images_masked = (images * (1 - masks).float()) + masks  # 用mask弄残
        outputs = self.generator(images_masked, 1 - masks)  # 第2个参数mask被ysy修改为1-masks，因为InpaintGenerator中pconv接受的mask的格式是源1洞0. 而mask本身是源0洞1                     # in: [rgb(3) + edge(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if self.config.ENABLE_D1: # and (dis_loss is not None):
            dis_loss.backward()
            self.dis_optimizer.step()
        #if gen_loss is not None:
        gen_loss.backward()
        self.gen_optimizer.step()


class RefineModel(BaseModel):
    def __init__(self, config):
        super(RefineModel, self).__init__('RefineModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        generator = RefineGenerator()
        discriminator = Discriminator(in_channels=3, use_sigmoid=not (
                    config.GAN_LOSS == 'hinge' or config.GAN_LOSS == 'wgan'))  # hinge 和 wgan 不用sigmoid
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator, config.GPU)

        l1_loss = nn.L1Loss()
        # perceptual_loss = PerceptualLoss()
        # style_loss = StyleLoss()
        perceptual_and_style_loss = PerceptualStyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)

        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        # self.add_module('perceptual_loss', perceptual_loss)
        # self.add_module('style_loss', style_loss)
        self.add_module('perceptual_and_style_loss', perceptual_and_style_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        if self.config.GAN_LOSS == 'wgan':  # wgan使用rmsprop
            self.gen_optimizer = optim.RMSprop(params=generator.parameters(),
                                               lr=float(config.LR))  # betas=(config.BETA1, config.BETA2))
            self.dis_optimizer = optim.RMSprop(params=discriminator.parameters(), lr=float(config.LR) * float(
                config.D2G_LR))  # ,???betas=(config.BETA1, config.BETA2))
        else:  # 默认使用adam
            self.gen_optimizer = optim.Adam(params=generator.parameters(), lr=float(config.LR),
                                            betas=(config.BETA1, config.BETA2))
            self.dis_optimizer = optim.Adam(params=discriminator.parameters(),
                                            lr=float(config.LR) * float(config.D2G_LR),
                                            betas=(config.BETA1, config.BETA2))

    # def process(self, images, edges, masks):
    def process(self, images_gt, images_in, masks):  # 在train和eval时调用process，test时不会   因为Process = 前向传播 + 获取损失， test时无需获取损失
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        # outputs = self(images_gt, edges, masks)
        outputs = self(images_gt, images_in, masks)  # 喂给generator
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = images_gt
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        # gen_l1_loss = self.l1_loss(outputs, images_gt) / torch.mean(masks)  # original
        gen_l1_loss = 6 * torch.mean(torch.abs(outputs - images_gt) * masks) + torch.mean(torch.abs(outputs - images_gt) * (1 - masks))  # 6 hole_loss + 1 valid_loss
        gen_loss += gen_l1_loss * self.config.L1_LOSS_WEIGHT

        # # generator perceptual loss
        # gen_content_loss = self.perceptual_loss(outputs, images_gt)
        # gen_loss += gen_content_loss * self.config.PERCEP_LOSS_WEIGHT
        #
        # # generator style loss
        # gen_style_loss = self.style_loss(outputs * masks, images_gt * masks)
        # gen_loss += gen_style_loss * self.config.STYLE_LOSS_WEIGHT

        gen_content_loss,gen_style_loss= self.perceptual_and_style_loss(outputs, images_gt)
        gen_loss += gen_content_loss * self.config.PERCEP_LOSS_WEIGHT
        gen_loss += gen_style_loss * self.config.STYLE_LOSS_WEIGHT

        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images_gt, images_in, masks):  # train, eval, test 最后都会到这里
        # images_masked = (images * (1 - masks).float()) + masks  # 取消弄残，因为第2阶段的输入不该带空洞
        images_in = (F.interpolate(images_in, scale_factor=2, mode='bilinear', align_corners=True) * masks) + (images_gt * (1 - masks))  # 低分辨率图 上采样后 和高分辨率gt融合
        outputs = self.generator(images_in, 1-masks)  # mask本身是源0洞1
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward(retain_graph = True)
        gen_loss.backward()

        self.dis_optimizer.step()
        self.gen_optimizer.step()

        # orig impl.
        # dis_loss.backward()
        # self.dis_optimizer.step()
        #
        # gen_loss.backward(retain_graph = True)
        # self.gen_optimizer.step()

