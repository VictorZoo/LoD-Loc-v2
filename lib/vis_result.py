import PIL.Image as Image
import glob
import numpy as np
import os
img2_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/images/render_upright/render4/UAV_night_sequence1/"
img1_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/images/images_upright/query/UAV/night/sequence1/"
save_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/compare_vis/sequence8/"

if not os.path.exists(save_path):
    os.mkdir(save_path)
img_list = glob.glob(img1_path + "/*.jpg")
img_list = np.sort(img_list)

for filename in img_list:
    img_name = filename.split('/')[-1].split('.')[0]
    img1_name =img1_path + img_name + '.jpg'
    img1 = Image.open(img1_name)
    img2_name =img2_path + img_name + '.png'
    img2 = Image.open(img2_name )

    img = Image.new(img1.mode, img1.size)

    # 获取图片的宽度和高度

    width, height = img1.size

    # 将第一张图片的左上角和右下角复制到新图片的对应位置

    img.paste(img1.crop((0, 0, width // 2, height // 2)), (0, 0))

    img.paste(img1.crop((width // 2, height // 2, width, height)), (width // 2, height // 2))

    # 将第二张图片的右上角和左下角复制到新图片的对应位置

    img.paste(img2.crop((width // 2, 0, width, height // 2)), (width // 2, 0))

    img.paste(img2.crop((0, height // 2, width // 2, height)), (0, height // 2))

    # 保存合成后的图片

    path = save_path + img_name + '.png'

    img.save(path)