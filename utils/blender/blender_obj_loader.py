import os
from pathlib import Path


def import_objs(blender_path, project_path, script_path, input_Objs, origin):
    """
    Import objects into Blender using a batch rendering script.

    Args:
        blender_path (str): Path to the Blender executable.
        project_path (str): Path to the .blend project file.
        script_path (str): Path to the Python script for rendering.
        input_Objs (str): Path to the input objects.
        origin (str): Path to the origin file or setting.
    """
    # Construct the command to run Blender with the specified script and project
    cmd = f"{blender_path} -b {project_path} -P {script_path} --batch {' '.join([project_path, input_Objs, origin])}"

    # Execute the command
    os.system(cmd)



def main(config):
    """
    Main function to set up paths and load Blender projects.

    Args:
        config (dict): Configuration dictionary containing paths and settings.
    """
    dataset = Path(config["render2loc"]["datasets"])
    blender_config = config["render2loc"]["blender"]
    
    # Construct paths for RGB and depth images
    rgb_path = str(dataset / blender_config["rgb_path"])
    depth_path = str(dataset / blender_config["depth_path"])
    
    # Retrieve paths for Python scripts used in Blender
    rgb_python_path = blender_config["python_importObjs_rgb_path"]
    depth_python_path = blender_config["python_importObjs_depth_path"]
    
    blender_path = blender_config["blender_path"]
    
    origin = str(dataset / blender_config["origin"])
    input_Objs = str(dataset / blender_config["input_recons"])

    print("Load obj models...")

    # Check if the RGB path exists and create it if not, then import objects
    if not os.path.exists(rgb_path):
        cmd = f"{blender_path} -b -P - --python-expr \"import bpy; bpy.ops.wm.read_factory_settings(); bpy.ops.wm.save_as_mainfile(filepath='{rgb_path}')\""
        os.system(cmd)
        
        print("Importing RGB objects...")
        import_objs(
            blender_path,
            rgb_path,
            rgb_python_path,
            input_Objs,
            origin
        )

    # Check if the depth path exists and create it if not, then import objects
    if not os.path.exists(depth_path):
        cmd = f"{blender_path} -b -P - --python-expr \"import bpy; bpy.ops.wm.read_factory_settings(); bpy.ops.wm.save_as_mainfile(filepath='{depth_path}')\""
        os.system(cmd)
        
        import_objs(
            blender_path,
            depth_path,
            depth_python_path,
            input_Objs,
            origin
        )


if __name__ == "__main__":
    # Assume 'config' is a predefined dictionary containing configuration information for rendering to location.
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
    