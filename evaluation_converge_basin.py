from gloc.models import features
import torch 
from gloc.models.refinement_model import DenseFeaturesRefiner
from types import SimpleNamespace
from dataclasses import dataclass
from path_configs import get_path_conf
from parse_args import parse_args

@dataclass
class DinoConf:
    clamp: float = -1
    level: int = 8
    pretrain: str = 'checkpoint-step=40000.ckpt'
    def get_str__conf(self):
        repr = f"_l{self.level}_cl{self.clamp}_pretrain{self.pretrain}"
        return repr
    
@dataclass
class DenseFeaturesConf:
    clamp: float = -1
    def get_str__conf(self):
        repr = f"_cl{self.clamp}"
        return repr

def get_feature_model(args):
    if model_name == 'Dinov2_contrast':
        conf = DinoConf(clamp=args.clamp_score, level=args.feat_level, pretrain=args.pretrain_model)
        feat_model = features.DinoFeatures_contrast(conf)
    else:
        raise NotImplementedError()
    
    if args.pretrain_model:
        assert args.pretrain_model is not None, "Please specify foundation model path."
        
        state_dict = torch.load(args.pretrain_model)
        new_state_dict = {}
        for key, value in state_dict['state_dict'].items():
            # 检查是否需要修改键名
            if key.startswith("model."):
                new_key = key.replace("model.", "")
            else:
                new_key = key
            new_state_dict[new_key] = value

        feat_model.load_state_dict(new_state_dict)
    if args.cuda:
        feat_model = feat_model.cuda()
    return feat_model

def get_class_model(model_class_name):
    if model_class_name == 'DenseFeatures':
        from gloc.models.refinement_model import DenseFeaturesRefiner
        model_class = DenseFeaturesRefiner
        conf = DenseFeaturesConf(clamp=args.clamp_score)
    return model_class, conf

if __name__ == '__main__':
    args = parse_args()

    model_class_name = 'DenseFeatures'
    model_name = 'Dinov2_contrast' # model name
    feat_level = [12] # if extract intermedia features
    pretrain_model = 'checkpoint-step=40000.ckpt' # pretrained model
    clamp_score = -1 # thresholded scoring function
    cuda = True
    args = {
        'model_name': model_name,
        'clamp_score': clamp_score,
        'pretrain_model': pretrain_model,
        'feat_level' : feat_level,
        "cuda": cuda
    }
    args = SimpleNamespace(**args)

    paths_conf = get_path_conf(args.colmap_res, args.mesh)
    # get model
    model_class, conf = get_class_model(model_class_name)
    feat_model = get_feature_model(args)
    model = model_class(conf, feat_model)
    if cuda:
        model = model.cuda()


