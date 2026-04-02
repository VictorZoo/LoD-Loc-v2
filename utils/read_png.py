import cv2
import numpy as np
# 读取PNG图片
def calculateCoefficients(zNear, zFar):
    a = zFar / (zFar - zNear)
    b = zFar * zNear / (zNear - zFar)
    return a, b
def inverseZBuffer(zBufferValue, a, b, nBits = 16):
    maxZBufferValue = nBits * 4096 - 1
    z = b / (zBufferValue / maxZBufferValue - a)
    return z

# 图像路径
image_path = 'your_image.png'

# 使用 cv2.imread() 读取图像
image = cv2.imread('/home/ubuntu/Documents/code/3DTilesRender-3DTilesRender_dev/python/depthImage-19.tiff', cv2.IMREAD_ANYDEPTH)

# 检查图像是否加载成功
if image is None:
    print("无法读取图像，请检查路径是否正确。")
else:
    # 打印图像的数据类型
    print(f"图像数据类型: {image.dtype}")

    # 检查图像是否为16位
    if image.dtype == np.float32 or image.dtype == np.float64:
        # 如果数据类型是float32或float64，可能是16位图像数据
        print("这可能是一个16位的图像。")
    elif image.dtype == np.uint16 or image.dtype == np.int16:
        print("这确实是一个16位的图像。")
    else:
        print("这不是一个16位的图像。")
# 确保图片已经被正确读取
if image is not None:
    # 计算并打印最大和最小像素值
    min_val = image.min()
    max_val = image.max()
    print(image)

    print(f'Minimum pixel value: {min_val}')
    print(f'Maximum pixel value: {max_val}')
else:
    print('Error: Image not found or unable to read the image.')
    
    
    
    
    
    
    
    
    
