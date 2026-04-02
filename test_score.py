from gloc.models.depth_anything_v2.dpt import DepthAnythingV2
import torch
import glob
import numpy as np
import os 
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_configs_depthAnything = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vitb'

def find_most_similar_batch(rgb_features, target_features):
        """_summary_

        Args:
            q_feats (np.array): shape 1 x C x H x W
            r_db_descriptors (np.array): shape N_cand x C x H x W

        Returns:
            torch.tensor : vector of shape (N_cand, ), score of each one
        """
        """
        找出每个RGB特征最相似的特征的batch序号
        rgb_features: [8, H, W]
        target_features: [72, H, W]
        返回: [8] (每个元素为最相似的特征的batch序号)
        """
        
        batch_size_rgb,_, H, W = rgb_features.shape
        batch_size_target,_, _, _ = target_features.shape

        # 归一化特征
        rgb_features_norm = F.normalize(rgb_features.view(batch_size_rgb, -1), p=2, dim=-1)
        target_features_norm = F.normalize(target_features.view(batch_size_target, -1), p=2, dim=-1)

        # 计算相似性矩阵
        similarity_matrix = torch.matmul(rgb_features_norm, target_features_norm.t())  # [8, 72]
        
        # 找出最相似的特征的batch序号
        most_similar_batch_indices = torch.argmax(similarity_matrix, dim=1)  # [8]

        return similarity_matrix.squeeze()# , most_similar_batch_indices

def get_model(pth=None):
    feat_model = DepthAnythingV2(**model_configs_depthAnything[encoder])
    feat_model.load_state_dict(torch.load(f'ckpt/depth_anything_v2_{encoder}.pth', map_location='cpu'))

    if pth:
        assert pth is not None, "Please specify foundation model path."
        # model_dict = feat_model.state_dict()
        
        state_dict = torch.load(pth)
        new_state_dict = {}
        for key, value in state_dict['state_dict'].items():
            # 检查是否需要修改键名
            if key.startswith("model.feature_extraction."):
                new_key = key.replace("model.feature_extraction.", "")
            else:
                new_key = key
            new_state_dict[new_key] = value
        # model_dict.update(state_dict.items())
        # model_dict.update(new_state_dict.items())
        feat_model.load_state_dict(new_state_dict)
    feat_model = feat_model.cuda()

    return feat_model

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = np.ascontiguousarray(image[:, :, ::-1])  # BGR to RGB
    return image

def process_image(image):
    image = (
        torch.from_numpy(np.ascontiguousarray(image))
        .permute(2, 0, 1)
        .float()
        .div_(255)
    )
    return image

def process_image_lod(image):
    image = (
        torch.from_numpy(np.ascontiguousarray(image))
        .unsqueeze(0) 
        .float()
        .div_(255)
        .repeat(3, 1, 1)
    )
    return image

def visualize_single_descriptor(descriptor, save_path="output.png", title=None):
    """
    可视化高维描述子张量，并保存图像.

    参数:
    descriptor (torch.Tensor): 形状为 [1, dim, H, W] 的描述子张量.
    save_path (str): 保存图像的路径.
    title (str): 图像的标题.

    返回:
    None
    """
    # 检查输入形状
    assert descriptor.shape[0] == 1, "输入的描述子张量的 batch 维度必须为 1"
    assert len(descriptor.shape) == 4, "输入的描述子张量形状必须为 [1, dim, H, W]"
    
    # 转换为 [H, W, dim]，并从 GPU 移动到 CPU（如果在 GPU 上）
    descriptor = descriptor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, dim]
    
    # 使用 PCA 将 dim 维降到 3 维
    pca = PCA(n_components=3)
    H, W, dim = descriptor.shape
    
    # 将特征向量降到 3 维
    flattened = descriptor.reshape(-1, dim)
    reduced = pca.fit_transform(flattened)
    reduced_descriptor = reduced.reshape(H, W, 3)
    
    # 将降维后的特征向量映射到 [0, 255] 的范围，并转换为 uint8 类型
    reduced_descriptor = (255 * (reduced_descriptor - np.min(reduced_descriptor)) / np.ptp(reduced_descriptor)).astype(np.uint8)
    
    # 可视化并保存
    plt.figure(figsize=(10, 5))
    plt.imshow(reduced_descriptor)
    plt.axis('off')
    
    # 如果有标题，设置标题
    if title:
        plt.title(title)

    # 保存图像
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"图像已保存到 {save_path}.")

def cal_score_sim(rgb_i, pred_i, gt_i, name):
    # RGB read
    image = read_image(rgb_i)
    image = process_image(image)

    # LoD read
    gt_render = read_image(gt_i, grayscale=True) 
    gt_render = process_image_lod(gt_render)

    pred_render = read_image(pred_i, grayscale=True) 
    pred_render = process_image_lod(pred_render)

    # Extract Feature
    with torch.no_grad():
        rgb_feat = fine_model(image.unsqueeze(0).cuda())
        gt_feat = fine_model(gt_render.unsqueeze(0).cuda())
        pred_feat = fine_model(pred_render.unsqueeze(0).cuda())

    rgb_save_pth = os.path.join('data/UAVD4L-LoD/inTraj/pred/vis/rgb', name)
    gt_save_pth = os.path.join('data/UAVD4L-LoD/inTraj/pred/vis/gt', name)
    pred_save_pth = os.path.join('data/UAVD4L-LoD/inTraj/pred/vis/pred', name)

    # Calculate Sim
    gt_score = find_most_similar_batch(rgb_feat[0], gt_feat[0])
    pred_score = find_most_similar_batch(rgb_feat[0], pred_feat[0])


    visualize_single_descriptor(rgb_feat[0], rgb_save_pth, title = 'RGB Fature')
    visualize_single_descriptor(gt_feat[0], gt_save_pth, title=f"Ground Truth Feature - Score: {gt_score:.2f}")
    visualize_single_descriptor(pred_feat[0], pred_save_pth, title=f"Predicted Feature - Score: {pred_score:.2f}")
    
    return gt_score, pred_score

def calculate_iou(img1, img2):
    """
    计算两个二值图像的交并比（IoU）。
    
    参数:
    img1 -- NumPy 数组, 形状为 (1, H, W) 或 (H, W) 的二值图像
    img2 -- NumPy 数组, 形状为 (1, H, W) 或 (H, W) 的二值图像
    
    返回:
    IoU -- float, 两个图像的交并比
    """
    # 如果输入图像形状为 (1, H, W)，则去掉第一个维度
    if len(img1.shape) == 3:
        img1 = img1.squeeze()
    if len(img2.shape) == 3:
        img2 = img2.squeeze()

    # 计算交集和并集
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()

    # 计算 IoU，处理并集为 0 的情况
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    else:
        return intersection / union
    
def cal_IoU(rgb_i, pred_i, gt_i):
    # RGB read
    image = read_image(rgb_i, grayscale=True)
    image = process_image_lod(image)

    # LoD read
    gt_render = read_image(gt_i, grayscale=True) 
    gt_render = process_image_lod(gt_render)

    pred_render = read_image(pred_i, grayscale=True) 
    pred_render = process_image_lod(pred_render)

    IoU_gt = calculate_iou(image, gt_render)
    IoU_pred = calculate_iou(image, pred_render)

    return IoU_gt, IoU_pred


if __name__ == "__main__":
    rgb_path = 'data/UAVD4L-LoD/inTraj/query_rgb'
    gt_render_path = 'data/UAVD4L-LoD/inTraj/gt_lod_render'
    pred_render_path = '/home/ubuntu/code/render2loc/data/poses/vis_pred'

    ckpt_pth = 'ckpt/Depthv2_maskTrain3.ckpt'

    fine_model = get_model(ckpt_pth)

    img_list = glob.glob(pred_render_path + "/*.png")
    # img_list = glob.glob(rgb_path + "/*.jpg")
    img_list = np.sort(img_list)
    count = 0
    for filename in img_list:
        pred_i = os.path.join(pred_render_path, filename.split('/')[-1])#.split('.')[0]+'.png')
        rgb_i = os.path.join(rgb_path, filename.split('/')[-1].split('.')[0]+'.JPG')
        gt_i = os.path.join(gt_render_path, filename.split('/')[-1])# .split('.')[0]+'.png')
        name = filename.split('/')[-1]
        # # Cal Sim
        gt_score, pred_score = cal_score_sim(rgb_i, pred_i, gt_i, name)
        if gt_score > pred_score:
            count += 1
        # print(gt_score > pred_score)
        # print(count)


        # Cal IoU
        # IoU_gt, IoU_pred = cal_IoU(rgb_i, pred_i, gt_i)
        # if IoU_gt > IoU_pred:
        #     count += 1
        # print(IoU_gt > IoU_pred)
        # print(IoU_gt, IoU_pred)
    print(count)

