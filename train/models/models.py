import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import resnet, vgg, inception, densenet
from ..utils.lib.nn import SynchronizedBatchNorm2d as BatchNorm2d
from .utils import get_upsampling_weight
from collections import OrderedDict

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10) * 100

        return acc

    def dice_coef(self, pred, gth, class_no=150):

        _, preds = torch.max(pred, dim=1)

        import numpy as np
        preds = preds.detach().to('cpu').numpy()
        gth = gth.detach().to('cpu').numpy()
        cnt = 0
        sub_dice = 0

        for cls in range(1, class_no):
            temp = gth[gth == cls]
            if len(temp):
                mask_gth = (gth == cls)
                mask_seg = (preds == cls)

                a_p = np.sum(mask_gth)
                # area of contour in label
                a_l = np.sum(mask_seg)
                # area of intersection
                a_pl = np.sum(mask_gth * mask_seg)
                sub_dice = sub_dice + 2. * a_pl / (a_p + a_l)
                cnt = cnt + 1

        if cnt==0:
            dice_coeff=0
        else:
            dice_coeff = sub_dice / cnt

        #print (dice_coeff)
        return dice_coeff

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        # training
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda")
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'].to(device), return_feature_maps=True))
            else:
                pred = self.decoder(self.encoder(feed_dict['img_data'].to(device), return_feature_maps=True))
            loss = self.crit(pred, feed_dict['seg_label'].to(device))
            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'].to(device))
                loss = loss + loss_deepsup * self.deep_sup_scale
            acc = self.pixel_acc(pred, feed_dict['seg_label'].to(device))
            dice = self.dice_coef(pred, feed_dict['seg_label'].to(device))
            s_outputs = nn.functional.softmax(pred)

            return loss, acc, s_outputs, dice
        # inference
        else:
            pred = self.decoder(self.encoder(feed_dict['img_data'].to(device), return_feature_maps=True), segSize=segSize)
            return pred

class ClassificationModuleBase(nn.Module):
    def __init__(self):
        super(ClassificationModuleBase, self).__init__()

    def cls_acc(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ClassificationModule(ClassificationModuleBase):
    def __init__(self, net_enc, crit, deep_sup_scale=None, incpt=False):
        super(ClassificationModule, self).__init__()
        self.encoder = net_enc
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale
        self.incpt = incpt

    def forward(self, feed_dict, *, segSize=None):
        # training
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda")
        #print (self.encoder)
        if segSize is None:
            ### pred output
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.encoder(feed_dict[0].to(device), return_feature_maps=False, _type='cls')
            elif self.incpt is not None:
                outs = self.encoder(feed_dict[0].to(device), return_feature_maps=False, _type='cls')
                if len(outs) == 2:
                    (pred, aux_logits) = outs
                else:
                    aux_logits = None
                    pred = outs
            else:
                pred = self.encoder(feed_dict[0].to(device), return_feature_maps=False, _type='cls')

            ### loss output
            if self.incpt:
                loss1 = self.crit(pred, feed_dict[1].to(device))
                loss2 = 0
                if aux_logits is not None:
                    loss2 = self.crit(aux_logits, feed_dict[1].to(device))
                loss = loss1 + 0.4*loss2
            else:
                loss = self.crit(pred, feed_dict[1].to(device))

            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict[1].to(device))
                loss = loss + loss_deepsup * self.deep_sup_scale

            ### acc, score
            acc = self.cls_acc(pred, feed_dict[1].to(device), topk=(1,2))
            s_outputs = nn.functional.softmax(pred, dim=1)

            return loss, acc, s_outputs, 0
        # inference
        else:
            pred = self.encoder(feed_dict[0].to(device), return_feature_maps=False, _type='cls')
            s_outputs = nn.functional.softmax(pred, dim=1)

            return pred, s_outputs


class ModelBuilder:
    # weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights='', num_class=3):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()

        if arch == 'vgg16fcn':
            orig_vggnet = vgg.__dict__['vgg16'](pretrained=pretrained)
            net_encoder = VGG16fcn(orig_vggnet)
        elif arch == 'vgg16cls':
            orig_vggnet = vgg.__dict__['vgg16'](pretrained=pretrained)
            net_encoder = VGG16(orig_vggnet)
            net_encoder.classifier[6] = nn.Linear(4096,2)
            d = 0
        elif arch == 'vgg19bncls':
            orig_vggnet = vgg.__dict__['vgg19_bn'](pretrained=False, num_classes=2)
            net_encoder = VGG19bn(orig_vggnet)
        elif arch == 'resnet50cls':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
            net_encoder.fc = nn.Linear(512 * 4, 3)
            d = 0
        elif arch == 'inceptionv3cls':
            orig_incept = inception.__dict__['inception_v3'](pretrained=pretrained)
            net_encoder = Inceptionv3(orig_incept)
            net_encoder.AuxLogits = inception.__dict__['InceptionAux'](768, 3)
            net_encoder.fc = nn.Linear(2048, 3)
            d = 0
        elif arch == 'densenet161cls':
            orig_densenet = densenet.__dict__['densenet161'](pretrained=pretrained)
            net_encoder = DENSENET161(orig_densenet)
            net_encoder.classifier = nn.Linear(2208, num_class)
            d = 0

        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            try:
                net_encoder.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            except:
                net_encoder.module.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'fcn8s':
            net_decoder = fcn8s(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=True)
        elif arch == 'aspp':
            net_decoder = ASPP(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        #elif arch == 'deeplabv3':
        #    #net_decoder = seg.deeplab.__dict__['deeplabv3']
        #    net_decoder = deeplabv3(
        #        num_class=num_class,
        #        fc_dim=fc_dim,
        #        use_softmax=use_softmax)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            try:
                net_decoder.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            except:
                net_decoder.module.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class DENSENET161(nn.Module):
    def __init__(self, orig_densenet):
        super(DENSENET161, self).__init__()

        self.features = orig_densenet.features
        self.classifier = orig_densenet.classifier

    def forward(self, x, return_feature_maps=False, _type='seg'):
        features = self.features(x)

        if _type == 'seg':
            if return_feature_maps:
                return [x]
            else:
                x = F.relu(features, inplace=True)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return [x]
        elif _type == 'cls':
            if return_feature_maps:
                return [x]
            else:
                x = F.relu(features, inplace=True)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        last_size = int(math.ceil(512 / 32))
        self.avgpool = nn.AvgPool2d(last_size, stride=1)
        self.fc = nn.Linear(512 * 4, 1000)

    def forward(self, x, return_feature_maps=False, _type='seg'):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);


        if _type == 'seg':
            if return_feature_maps:
                return conv_out
            return [x]
        elif _type == 'cls':
            if return_feature_maps:
                return conv_out
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

class VGG16fcn(nn.Module):
    def __init__(self, orig_vggnet):
        super(VGG16fcn, self).__init__()

        self.features = orig_vggnet.features

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            return [x]
        else:
            return [x]

class VGG16(nn.Module):
    def __init__(self, orig_vggnet):
        super(VGG16, self).__init__()

        self.features = orig_vggnet.features
        self.avgpool = orig_vggnet.avgpool
        self.classifier = orig_vggnet.classifier

    def forward(self, x, return_feature_maps=False, _type='seg'):
        x = self.features(x)

        if _type == 'seg':
            if return_feature_maps:
                return [x]
            else:
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return [x]
        elif _type == 'cls':
            if return_feature_maps:
                return [x]
            else:
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

class VGG19bn(nn.Module):
    def __init__(self, orig_vggnet):
        super(VGG19bn, self).__init__()

        self.features = orig_vggnet.features
        self.avgpool = orig_vggnet.avgpool
        self.classifier = orig_vggnet.classifier

    def forward(self, x, return_feature_maps=False, _type='seg'):
        x = self.features(x)

        if _type == 'seg':
            if return_feature_maps:
                return [x]
            else:
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return [x]
        elif _type == 'cls':
            if return_feature_maps:
                return [x]
            else:
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

class Inceptionv3(nn.Module):
    def __init__(self, orig_incept):
        super(Inceptionv3, self).__init__()

        init_weights = True

        self.aux_logits = orig_incept.aux_logits
        self.transform_input = orig_incept.transform_input
        self.Conv2d_1a_3x3 = orig_incept.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = orig_incept.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = orig_incept.Conv2d_2b_3x3
        self.maxpool1 = orig_incept.maxpool1
        self.Conv2d_3b_1x1 = orig_incept.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = orig_incept.Conv2d_4a_3x3
        self.maxpool2 = orig_incept.maxpool2
        self.Mixed_5b = orig_incept.Mixed_5b
        self.Mixed_5c = orig_incept.Mixed_5c
        self.Mixed_5d = orig_incept.Mixed_5d
        self.Mixed_6a = orig_incept.Mixed_6a
        self.Mixed_6b = orig_incept.Mixed_6b
        self.Mixed_6c = orig_incept.Mixed_6c
        self.Mixed_6d = orig_incept.Mixed_6d
        self.Mixed_6e = orig_incept.Mixed_6e

        self.AuxLogits = None
        if orig_incept.aux_logits:
            self.AuxLogits = orig_incept.AuxLogits

        self.Mixed_7a = orig_incept.Mixed_7a
        self.Mixed_7b = orig_incept.Mixed_7b
        self.Mixed_7c = orig_incept.Mixed_7c
        self.avgpool = orig_incept.avgpool
        self.dropout = orig_incept.dropout
        self.fc = orig_incept.fc

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        if self.training and self.aux_logits:
            return inception.InceptionOutputs(x, aux)
        else:
            return x  ## type: ignore[return-value]

    def forward(self, x, return_feature_maps=False, _type='seg'):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                print ("Scripted Inception3 always returns Inception3 Tuple")
            return inception.InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling
class ASPP(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 use_softmax=False, rate=(12, 24, 36)):
        super(ASPP, self).__init__()
        self.use_softmax = use_softmax
        self.out_channels = 256

        self.moduels = []
        self.moduels.append(nn.Sequential(
            nn.Conv2d(fc_dim, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()))
        for dilation in rate:
            self.moduels.append(nn.Sequential(
            nn.Conv2d(fc_dim, self.out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU())
            )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fc_dim, self.out_channels, 1, bias=False),
            BatchNorm2d(self.out_channels),
            nn.ReLU())
        self.convs = nn.ModuleList(self.moduels)

        self.prj = nn.Sequential(
            nn.Conv2d(5*self.out_channels, self.out_channels, 1, bias=False),
            BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
            BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, num_class, 1)
        )

    def forward(self, conv_out, segSize=None):
        x = conv_out[-1]
        input_size = x.size()
        #res = [x]
        res=[]
        for conv in self.convs:
            """
            res.append(nn.functional.interpolate(
                conv(x),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
            """
            res.append(conv(x))
        res.append(nn.functional.interpolate(
            self.pool(x),
            (input_size[2], input_size[3]),
            mode='bilinear', align_corners=False))
        #print (x)
        res = torch.cat(res, dim=1)
        x = self.prj(res)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x

class fcn8s(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False):
        super().__init__()
        self.use_softmax = use_softmax

        n_class = 21
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._upscore8 = nn.Conv2d(n_class, num_class, 1, 1, 0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, segSize=None):
        def dbg(cfg, out):
            if cfg: print(out.size())
        dbg_cfg = 0
        _x = x[-1]

        out = x[-1]

        dbg(dbg_cfg, out)
        out = self.relu1_1(self.conv1_1(out))
        out = self.relu1_2(self.conv1_2(out))
        out = self.pool1(out)

        dbg(dbg_cfg, out)
        out = self.relu2_1(self.conv2_1(out))
        out = self.relu2_2(self.conv2_2(out))
        out = self.pool2(out)

        dbg(dbg_cfg, out)
        out = self.relu3_1(self.conv3_1(out))
        out = self.relu3_2(self.conv3_2(out))
        out = self.relu3_3(self.conv3_3(out))
        out = self.pool3(out)
        pool3 = out  # 1/8

        dbg(dbg_cfg, out)
        out = self.relu4_1(self.conv4_1(out))
        out = self.relu4_2(self.conv4_2(out))
        out = self.relu4_3(self.conv4_3(out))
        out = self.pool4(out)
        pool4 = out  # 1/16

        dbg(dbg_cfg, out)
        out = self.relu5_1(self.conv5_1(out))
        out = self.relu5_2(self.conv5_2(out))
        out = self.relu5_3(self.conv5_3(out))
        out = self.pool5(out)

        dbg(dbg_cfg, out)
        out = self.relu6(self.fc6(out))
        out = self.drop6(out)

        dbg(dbg_cfg, out)
        out = self.relu7(self.fc7(out))
        out = self.drop7(out)

        dbg(dbg_cfg, out)
        out = self.score_fr(out)
        out = self.upscore2(out)
        upscore2 = out  # 1/16

        dbg(dbg_cfg, out)
        out = self.score_pool4(pool4)
        out = out[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = out  # 1/16

        dbg(dbg_cfg, out)
        out = upscore2 + score_pool4c  # 1/16
        out = self.upscore_pool4(out)
        upscore_pool4 = out  # 1/8

        dbg(dbg_cfg, out)
        out = self.score_pool3(pool3)
        out = out[:, :,
            9:9 + upscore_pool4.size()[2],
            9:9 + upscore_pool4.size()[3]]
        score_pool3c = out  # 1/8

        dbg(dbg_cfg, out)
        out = upscore_pool4 + score_pool3c  # 1/8

        dbg(dbg_cfg, out)
        out = self.upscore8(out)
        out = out[:, :, 31:31 + _x.size()[2], 31:31 + _x.size()[3]].contiguous()

        dbg(dbg_cfg, out)
        out = self._upscore8(out)

        dbg(dbg_cfg, out)
        if self.use_softmax:
            out = nn.functional.log_softmax(out, dim=1)

        return out
"""
class deeplabv3(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096, use_softmax=False):
        super(deeplabv3, self).__init__()
        #self.classifier = classifier
        self.classifier = seg.deeplab.__dict__['deeplabhead'](2048,2)
        self.aux_classifier = None


    def forward(self, x):
        input_shape = list(x[0].shape[-2:])
        # contract: features is a dict of tensors

        #print (x)
        x = self.classifier(x)
        x = nn.functional.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)


        return x
"""

