from pathlib import Path
from lib.read_model import parse_image_lists, parse_db_intrinsic_list, parse_pose_list
from lib import get_lon_lat
def get_pairs_imagepath(pairs, render_path: Path, image_path: Path, engine = 'blender'):
    """
    Generate a dictionary of image paths for query and render images based on pairs.

    Args:
        pairs (dict): Dictionary containing query names and corresponding list of render names.
        render_path (Path): Path to the directory where render images are stored.
        image_path (Path): Path to the directory where query images are stored.

    Returns:
        dict: Dictionary with query image paths as keys and lists of tuples containing
              render image paths and corresponding EXR file paths as values.
    """
    render_dir = {}
    for query_name, imgr_name_list in pairs.items():
        renders = []
        if 'query' not in query_name:
            query_name = 'query/' + query_name
        imgq_pth = image_path / query_name  # Path to the query image
        
        for imgr_name in imgr_name_list:
            imgr = imgr_name.split('/')[-1].split('.')[0]  # Extract image name from the render name
            
            if engine == 'blender':
                imgr_pth = str(render_path / (imgr + '.jpg'))  # Path to the render image
                exrr_pth = str(render_path / (imgr + '0001.exr'))# Path to the corresponding EXR file
            elif engine == 'osg':
                imgr_pth = str(render_path / (imgr + '.png'))  # Path to the render image
                exrr_pth = str(render_path / (imgr + '.tiff'))# Path to the corresponding EXR file
                
            # Create a tuple of the render image path and the EXR file path
            render = [imgr_pth, exrr_pth]
            
            renders.append(render)  # Add the tuple to the list of renders
        
        # Use the string representation of the query image path as the key
        render_dir[str(imgq_pth)] = renders
    
    return render_dir

def get_render_candidates(renders, queries):
    """
    Creates a dictionary of pairs where each query image is paired with a corresponding render image.

    Args:
        renders (list): A list of render image identifiers.
        queries (list): A list of query image identifiers.

    Returns:
        dict: A dictionary with query image identifiers as keys and lists containing
              a single render image identifier as values.
    """
    pairs = {}
    for query_name in queries:
        render_candidate = []
        query = (query_name.split('/')[-1]).split('.')[0]
        for render_name in renders:
            if query in render_name:
                render_candidate.append(render_name)
        pairs[query_name] = render_candidate
    return pairs
    
    
    # pairs = {}
    # for i in range(len(queries)):
    #     # For each query, create a key in the dictionary with the query name
    #     # and assign a list containing the corresponding render name as the value
    #     pairs[queries[i]] = [renders[i]]

    # return pairs
 
def main(image_dir: Path, render_dir: Path, query_camera: Path, render_camera: Path, render_extrinsics: Path, data, engine = 'blender'):
    """
    Main function to process camera data and generate image pairs for rendering.

    Args:
        image_dir (Path): Path to the directory containing query images.
        render_dir (Path): Path to the directory with rendered images.
        query_camera (Path): Path to the file containing query camera data.
        render_camera (Path): Path to the file containing render camera data.
        render_extrinsics (Path): Path to the file containing render extrinsics data.
        data (dict): Dictionary to store and return processed data.
        iter (int): Iteration number or parameter for processing.

    Returns:
        dict: Updated dictionary with rendering data and image pairs.
    """

    # Ensure that the specified camera and extrinsics files exist
    assert render_camera.exists(), "Render camera file does not exist."
    assert render_extrinsics.exists(), "Render extrinsics file does not exist."
    assert query_camera.exists(), "Query camera file does not exist."

    # Parse the render extrinsics file to get the render poses
        
    render_pose = parse_pose_list(render_extrinsics)
    render_name = [key for key, _ in render_pose.items()] 
    # Parse the query camera file to get the query camera intrinsics and names
    K_q = parse_image_lists(query_camera, with_intrinsics=True, simple_name=False)
    query_name = [key for key, _ in K_q]

    # Parse the render camera file to get the render camera intrinsics
    K_render = parse_db_intrinsic_list(render_camera)

    # Create pairs of render and query images based on their names
    pairs = get_render_candidates(render_name, query_name)

    # Get the full image paths for all pairs of images
    all_pairs_path = get_pairs_imagepath(pairs, render_dir, image_dir, engine=engine)

    render_pose1 = dict()
    for key, item in render_pose.items():
        render_pose1[key.split('.')[0]] = item

    # Update the data dictionary with the collected information
    data["queries"] = K_q  
    data["query_name"] = query_name
    data["render_intrinsics"] = K_render
    data["render_pose"] = render_pose1
    data["pairs"] = all_pairs_path

    return data