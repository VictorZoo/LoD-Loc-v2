import sys
from pathlib import Path
from collections import OrderedDict
import torch
from torch import nn
from torchvision.models import resnet18, resnet50

from gloc.models.layers import L2Norm
import math
from ..backbone_side.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class BaseFeaturesClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()

    def forward(self, x):
        """
        To be used by subclasses, each specifying their own `self.model`
        Args:
            x (torch.tensor): batch of images shape Bx3xHxW
        Returns:
            torch.tensor: Features maps of shape BxDxHxW
        """
        return self.model(x)


class CosplaceFeatures(BaseFeaturesClass):
    def __init__(self, model_name):
        super().__init__()
        if 'r18' in model_name:
            arch = 'ResNet18'
        else: # 'r50' in model_name
            arch = 'ResNet50'
        # FC dim set to 512 as a placeholder, it will be truncated anyway before the last FC
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", 
                    backbone=arch, fc_output_dim=512)
                
        backbone = model.backbone    
        if '_l1' in model_name:
            backbone = backbone[:-3]
        elif '_l2' in model_name:
            backbone = backbone[:-2]
        elif '_l3' in model_name:
            backbone = backbone[:-1]
        
        self.model = backbone.eval()
        

class ResnetFeatures(BaseFeaturesClass):
    def __init__(self, model_name):
        super().__init__()        
        
        if model_name.startswith('resnet18'):
            model = resnet18(weights='DEFAULT')
        elif model_name.startswith('resnet50'):
            model = resnet50(weights='DEFAULT')
        else:
            raise NotImplementedError
        
        layers = list(model.children())[:-2]  # Remove avg pooling and FC layer
        backbone = torch.nn.Sequential(*layers)

        if '_l1' in model_name:
            backbone = backbone[:-3]
        elif '_l2' in model_name:
            backbone = backbone[:-2]
        elif '_l3' in model_name:
            backbone = backbone[:-1]
        
        self.model = backbone.eval()
        

class AlexnetFeatures(BaseFeaturesClass):
    def __init__(self, model_name):
        super().__init__()        
        
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        backbone = model.features
        
        if '_l1' in model_name:
            backbone = backbone[:4]
        elif '_l2' in model_name:
            backbone = backbone[:7]
        elif '_l3' in model_name:
            backbone = backbone[:9]
        
        self.model = backbone.eval()


class DinoFeatures(BaseFeaturesClass):
    def __init__(self, conf):
        super().__init__() 
        self.conf = conf
        self.clamp  = conf.clamp
        self.norm = L2Norm()
        self.conf.bs = 32
        self.feat_level = conf.level[0]

        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.ref_model = dinov2_vits14

    # override
    def forward(self, x):
        desc = self.ref_model.get_intermediate_layers(x, n=self.feat_level, reshape=True)[-1]
        # desc2 = self.ref_model.forward_features(x)
        desc = self.norm(desc)

        return desc


def initial_parameter(model):
    
    # Freeze parameters except adapter
    for name, param in model.backbone.named_parameters():
        if ("adapter" not in name) and ("prompt" not in name)  and ("lora" not in name):

            param.requires_grad = True #False True
        else:
            print(name)
    
        ## initialize Adapter
    for n, m in model.named_modules():
        if 'adapter' in n:
            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
            for n2, m2 in m.named_modules():
                if 'conv' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)

def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)  
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
    
class FoundationVPRNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self,pretrained_foundation = False, foundation_model_path = None):
        super().__init__()
        self.backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())
        self.norm = L2Norm()
        # encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1, batch_first=False)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) 
        # self.classifier = ClassBlock(input_dim=768, class_num=opt.nclasses, droprate=0.5,num_bottleneck=768)

    def forward(self, x, masks=None):
        x = self.backbone.get_intermediate_layers(x, n=12, reshape=True)[-1]
        # x = self.backbone(x)
        
        return x
        
    def forward_cnn1(self, x):
        # 修改多尺度融合（feature map）
        x_output = self.backbone(x) 
        x = x_output["x_norm_patchtokens"]
        x0_cls = x_output["x_norm_clstoken"]
        x1_cls = x[:, 256,:]
        x2_cls = x[:, 273, :]
        x3_cls = x[:, 278, :]


        x0 = x[:, :256,:]
        x1 = x[:, 257:273, :]
        x2 = x[:, 274:278, :]
        x3 = x[:, 279:280, :]
        B,P,D = x0.shape
        W = H = int(math.sqrt(P))
        x0, x1, x2, x3 = self.aggregation(x0.view(B,W,H,D).permute(0,3,1,2)), self.aggregation(x1.view(B,W//4,H//4,D).permute(0,3,1,2)), self.aggregation(x2.view(B,W//8,H//8,D).permute(0,3,1,2)), self.aggregation(x3.view(B,W//16,H//16,D).permute(0,3,1,2))
        x = [i.unsqueeze(1) for i in [x0, x1, x2, x3]]
        x = torch.cat(x, dim=1)

        x = self.encoder(x).view(B,4*D)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

class DinoFeatures_contrast(BaseFeaturesClass):
    def __init__(self, conf):
        super().__init__() 
        self.conf = conf
        self.clamp  = conf.clamp
        self.norm = L2Norm()
        self.conf.bs = 32
        self.feat_level = conf.level[0]
        self.foundation_model_path = conf.pretrain
        self.patch_size = 14

        self.feature_extraction = FoundationVPRNet(pretrained_foundation = False, foundation_model_path = self.foundation_model_path)  #改
        initial_parameter(self.feature_extraction)

    # override
    def forward(self, x):
        # desc = self.ref_model.get_intermediate_layers(x, n=self.feat_level, reshape=True, norm=False)[-1]
        # # desc2 = self.ref_model.forward_features(x)
        # desc = self.norm(desc)
        # B, _, w, h = x.shape
        features_dict = self.feature_extraction.backbone.forward_features(x)
        x = features_dict['x_prenorm']

        # desc = self.feature_extraction.backbone.get_intermediate_layers(x, n=self.feat_level, reshape=True, norm=False)[-1]
        # x = self.norm(desc)
        return x
    


class RomaFeatures(BaseFeaturesClass):
    weight_urls = {
        "roma": {
            "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
            "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
        },
        "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
    }

    def __init__(self, conf):
        super().__init__() 
        sys.path.append(str(Path(__file__).parent.joinpath('third_party/RoMa')))
        from roma.models.encoders import CNNandDinov2

        self.conf = conf
        weights = torch.hub.load_state_dict_from_url(self.weight_urls["roma"]["outdoor"])
        dinov2_weights = torch.hub.load_state_dict_from_url(self.weight_urls["dinov2"])

        ww = OrderedDict({k.replace('encoder.', ''): v for (k, v) in weights.items() if k.startswith('encoder')  })
        encoder = CNNandDinov2(
            cnn_kwargs = dict(
                pretrained=False,
                amp = True),
            amp = True,
            use_vgg = True,
            dinov2_weights = dinov2_weights
        )
        encoder.load_state_dict(ww)
        
        self.ref_model = encoder.cnn
        self.clamp  = conf.clamp
        self.scale_n = conf.scale_n
        self.norm = L2Norm()
        self.conf.bs = 16
        self.feat_level = conf.level[0]

    # override
    def forward(self, x):
        f_pyramid = self.ref_model(x)
        fmaps = f_pyramid[self.feat_level]

        if self.scale_n != -1:
            # optionally scale down fmaps
            nh, nw = tuple(fmaps.shape[-2:])
            half = nn.AdaptiveAvgPool2d((nh // self.scale_n, nw // self.scale_n))
            fmaps = half(fmaps)

        desc = self.norm(fmaps)
        return desc
