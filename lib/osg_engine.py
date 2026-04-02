import os
from pathlib import Path

def blender_command(
        osg_path,
        model_path,
        pose_path,
        save_path,
        window
):
    """
    Run the Blender engine to render images and process them with a script.

    Args:
        osg_path (str): Path to the Blender executable.
        project_path (str): Path to the .blend project file.
        script_path (str): Path to the Python script for rendering.
        origin (str): Path to the origin file or setting.
        sensor_height (float): The height of the sensor in millimeters.
        sensor_width (float): The width of the sensor in millimeters.
        f_mm (float): The focal length of the camera in millimeters.
        intrinsics_path (Path): Path to store camera intrinsics in COLMAP format.
        extrinsics_path (Path): Path to store camera extrinsics in COLMAP format.
        image_save_path (Path): Path to save the rendered images.
    ATTENTION: f_mm = 0: use focal length in intrinsics file, else use f_mm, sensor_height, sensor_width
    """
    # Construct the command to run Blender with the specified script and project
    cmd = '{} --path {} --pose_path {} --save_path {} --window {} --save'.format(
        osg_path,
        model_path,
        pose_path,
        save_path,
        window 
    )
    # Execute the command
    print(cmd)
    os.system(cmd)
    
def main(config):
    """
    Main function to set up paths and start the rendering process.

    Args:
        config (dict): Configuration dictionary containing paths and settings.
        intrinsics_path (Path): Path to store camera intrinsics.
        extrinsics_path (Path): Path to store camera extrinsics.
        img_save_path (Path): Path to save the rendered images.
    """
    dataset = Path(config["render2loc"]["datasets"])
    osg_config = config["render2loc"]["osg"]
    
    # Construct paths for model
    model_path = osg_config["path"]
    
    # Construct paths for poses
    pose_path = dataset / osg_config["pose_path"] 
    
    # Construct save paths for RGB and depht images
    save_path = dataset / osg_config["save_path"] 
          
    # Retrieve the path to the OSG executable
    osg_path = osg_config["osg_path"]
    
    # Retrieve image size
    # window = osg_config["window"]
    window = "800 600"
    
    print("Rendering images...")

    # Render RGB and depth images using the Blender engine
    blender_command(
        osg_path,
        str(model_path),
        str(pose_path),
        str(save_path),
        window
    )


if __name__ == "__main__":

    config = {
    "render2loc": {
        "datasets": "/home/ubuntu/Documents/code/Render2loc/datasets/demo8",
        "osg": {
            "path": "http://localhost:8080/Scene/Production_6.json",
            "pose_path": "gt_pose/gt_pose1.txt",
            "save_path": "images/render_upright",
            "osg_path": "/home/ubuntu/Documents/code/Render2loc/osg/build/ModelRenderScene",
            # "window": "800 600"
        }
    }
    }
    main(config)
    