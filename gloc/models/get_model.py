import logging
import sys
from dataclasses import dataclass
import torch
from torch import nn
import os
from torchvision.models import resnet18, resnet50

from gloc.models.layers import L2Norm, FlattenFeatureMaps
from gloc.models import features
from ..module import GenericModule
from .depth_anything_v2.dpt import DepthAnythingV2

model_configs_depthAnything = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def get_retrieval_model(model_name, cuda=True, eval=True):
    if model_name.startswith('cosplace'):
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", 
                    backbone="ResNet18", fc_output_dim=512)    
    else:
        raise NotImplementedError()
    
    if cuda:
        model = model.cuda()
    if eval:
        model = model.eval()

    return model


def get_feature_model(args, model_name, cuda=True):
    if model_name.startswith('cosplace'):
        feat_model = features.CosplaceFeatures(model_name)
    
    elif model_name.startswith('resnet'):
        feat_model = features.ResnetFeatures(model_name)

    elif model_name.startswith('alexnet'):
        feat_model = features.AlexnetFeatures(model_name)

    elif model_name == 'Dinov2':
        conf = DinoConf(clamp=args.clamp_score, level=args.feat_level)
        feat_model = features.DinoFeatures(conf)
    
    elif model_name == 'Roma':
        conf = RomaConf(clamp=args.clamp_score, level=args.feat_level, scale_n=args.scale_fmaps)
        feat_model = features.RomaFeatures(conf)

    elif model_name == 'Dinov2_contrast':
        conf = DinoConf(clamp=args.clamp_score, level=args.feat_level, pretrain=args.pretrain)
        feat_model = features.DinoFeatures_contrast(conf)

    elif model_name == 'DepthV2':
        # conf = DinoConf(clamp=args.clamp_score, level=args.feat_level, pretrain=args.pretrain)
        feat_model = DepthAnythingV2(**model_configs_depthAnything[args.encoder])
        # feat_model.load_state_dict(torch.load(f'ckpt/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    else:
        raise NotImplementedError()

    if cuda:
        feat_model = feat_model.cuda()

    return feat_model


def get_ref_model(args, cuda=True):
    model_name = args.ref_model
    feat_model = get_feature_model(args, args.feat_model, cuda)

    if model_name == 'DenseFeatures':
        from gloc.models.refinement_model import DenseFeaturesRefiner
        model_class = DenseFeaturesRefiner
        conf = DenseFeaturesConf(clamp=args.clamp_score)
        
    else:
        raise NotImplementedError()

    # if args.pretrain:
    #     assert args.pretrain is not None, "Please specify foundation model path."
    #     # model_dict = feat_model.state_dict()
        
    #     state_dict = torch.load(args.pretrain)
    #     new_state_dict = {}
    #     for key, value in state_dict['state_dict'].items():
    #         # 检查是否需要修改键名
    #         if key.startswith("model.feature_extraction."):
    #             new_key = key.replace("model.feature_extraction.", "")
    #         else:
    #             new_key = key
    #         new_state_dict[new_key] = value
    #     # model_dict.update(state_dict.items())
    #     # model_dict.update(new_state_dict.items())
    #     feat_model.load_state_dict(new_state_dict)

    model = model_class(conf, feat_model)
    if cuda:
        model = model.cuda()

    return model


@dataclass
class DenseFeaturesConf:
    clamp: float = -1
    def get_str__conf(self):
        repr = f"_cl{self.clamp}"
        return repr


@dataclass
class DinoConf:
    clamp: float = -1
    level: int = 8
    pretrain: str = 'checkpoint-step=40000.ckpt'
    def get_str__conf(self):
        repr = f"_l{self.level}_cl{self.clamp}_pretrain{self.pretrain}"
        return repr

@dataclass
class RomaConf:
    clamp: float = -1
    level: int = 4 # 1 2 4 8
    # pool feature maps to 1/n
    scale_n: int = -1
    def get_str__conf(self):
        repr = f"_l{self.level}_sn{self.scale_n}_cl{self.clamp}"
        return repr
