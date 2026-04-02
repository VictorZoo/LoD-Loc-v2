from pathlib import Path
import numpy as np
import cv2
from ModelRenderScene import ModelRenderScene

class RenderImageProcessor:
    def __init__(self, config):
        self.config = config
        self.osg_config = self.config["render2loc"]["osg"]
        # self.renderer = self._initialize_renderer()
        # self.renderer = self._initialize_renderer_obj()
        self.renderer = self._initialize_renderer_obj_fxfy()
        self._delay(self.osg_config)

    def _initialize_renderer(self):
        # Construct paths for model
        # 55.24912344,
        #     1.33333
        model_path = self.osg_config["model_path"]
        render_camera = self.config['render2loc']['render_camera']
        view_width, view_height = int(render_camera[0]), int(render_camera[1])
        self.fovy, aspectRatio = render_camera[-2], render_camera[-1]

        return ModelRenderScene(model_path, view_width, view_height, self.fovy, aspectRatio)
    
    def _initialize_renderer_obj(self):
        # Construct paths for model
        # 55.24912344,
        #     1.33333
        model_path = self.osg_config["model_path"]
        render_camera = self.config['render2loc']['render_camera']
        view_width, view_height = int(render_camera[0]), int(render_camera[1])
        self.fovy, aspectRatio = render_camera[-2], render_camera[-1]

        srsCode = 'EPSG:4547'
        x, y, z = 0., 0., 0. # obj_noTrans的時候沒用上
        # breakpoint()
        return ModelRenderScene(model_path, view_width, view_height, self.fovy, aspectRatio, srsCode, x, y, z)
    
    def _initialize_renderer_obj_fxfy(self):
        # Construct paths for model
        # 55.24912344,
        #     1.33333
        model_path = self.osg_config["model_path"]
        render_camera = self.config['render2loc']['render_camera']
        view_width, view_height = int(render_camera[0]), int(render_camera[1])
        self.fx, self.fy, self.cx, self.cy = render_camera[2], render_camera[3], render_camera[4], render_camera[5]

        srsCode = 'EPSG:4547'
        x, y, z = 0., 0., 0. # obj_noTrans的時候沒用上
        # breakpoint()  #改
        return ModelRenderScene(model_path, view_width, view_height, self.fx, self.fy, self.cx, self.cy, srsCode, x, y, z)
    
    def _delay(self, config):
        initTrans = config["init_trans"]
        initRot = config["init_rot"]
        # initTrans = [112.9955334820025, 28.29144220974383, 68.32987455979851]
        # initRot = [-125.24210197263284+180, 0.2695992071232557, 25.048717580689583]
        for i in range(100):
            self.update_pose(initTrans, initRot, fovy = None)
    
    def update_pose(self, Trans, Rot, fovy = None):
        if fovy is not None:
            self.fovy = fovy
        self.renderer.updateViewPoint(Trans, Rot)
        # self.renderer.nextFrame(self.fovy)  
        self.renderer.nextFrame(self.fx, self.fy, self.cx, self.cy)  
    def get_color_image(self):
        colorImgMat = np.array(self.renderer.getColorImage(), copy=False)
        colorImgMat = cv2.flip(colorImgMat, 0)
        colorImgMat = cv2.cvtColor(colorImgMat, cv2.COLOR_RGB2BGR)
        
        return colorImgMat
    
    def get_depth_image(self):
        depthImgMat = np.array(self.renderer.getDepthImage(), copy=False).squeeze()
        
        return depthImgMat
    
    def save_color_image(self, outputs):
        self.renderer.saveColorImage(outputs)
    
    
    
        
