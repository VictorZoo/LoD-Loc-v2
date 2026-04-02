import os
from pathlib import Path

def blender_command(
    blender_path, project_path, script_path,
    origin, sensor_height, sensor_width, f_mm,
    intrinsics_path, extrinsics_path, image_save_path
):
    """
    Run the Blender engine to render images and process them with a script.

    Args:
        blender_path (str): Path to the Blender executable.
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
    cmd = '{} -b {} -P {} -- {} {} {} {} {} {} {}'.format(
        blender_path,
        project_path,
        script_path,
        origin,
        sensor_height,
        sensor_width,
        f_mm,
        intrinsics_path,
        extrinsics_path,
        image_save_path,  
    )
    # Execute the command
    os.system(cmd)
    
def main(config, intrinsics_path, extrinsics_path, img_save_path):
    """
    Main function to set up paths and start the rendering process.

    Args:
        config (dict): Configuration dictionary containing paths and settings.
        intrinsics_path (Path): Path to store camera intrinsics.
        extrinsics_path (Path): Path to store camera extrinsics.
        img_save_path (Path): Path to save the rendered images.
    """
    dataset = Path(config["render2loc"]["datasets"])
    blender_config = config["blender"]
    
    # Construct paths for RGB and depth images
    rgb_path = dataset / blender_config["rgb_path"]
    depth_path = dataset / blender_config["depth_path"]
    
    # Retrieve paths for Python scripts used in Blender
    python_rgb_path = blender_config["python_rgb_path"]
    python_depth_path = blender_config["python_depth_path"]
    
    # Retrieve the path to the Blender executable
    blender_path = blender_config["blender_path"]
    
    # Retrieve the path to the origin setting
    origin = dataset / blender_config["origin"]
    
    # Retrieve camera sensor dimensions and focal length
    f_mm = blender_config["f_mm"]
    sensor_width = blender_config["sensor_width"]
    sensor_height = blender_config["sensor_height"]
    
    print("Rendering images...")

    # Render RGB and depth images using the Blender engine
    blender_command(
        blender_path,
        str(rgb_path),
        python_rgb_path,
        origin,
        sensor_height,
        sensor_width,
        f_mm,
        str(intrinsics_path),
        str(extrinsics_path),
        str(img_save_path),
    )
    
    blender_command(
        blender_path,
        str(depth_path),
        python_depth_path,
        origin,
        sensor_height,
        sensor_width,
        f_mm,
        str(intrinsics_path),
        str(extrinsics_path),
        str(img_save_path),
    )
    


if __name__ == "__main__":

    config = {
    "render2loc": {
        "datasets": "/path/to/datasets",
        "blender": {
            "rgb_path": "/path/to/rgb",
            "depth_path": "/path/to/depth",
            "python_importObjs_rgb_path": "/path/to/python/rgb",
            "python_importObjs_depth_path": "/path/to/python/depth",
            "blender_path": "/path/to/blender",
            "origin": "/path/to/origin",
            "input_recons": "/path/to/input/recons"
        }
    }
    }
    main(config)