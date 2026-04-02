import bpy
import os
import sys
import xmltodict
def delete_object(label):
    """
    Definition to delete an object from the scene.
    Parameters
    ----------
    label          : str
                     String that identifies the object to be deleted.
    """
    bpy.data.objects.remove(bpy.context.scene.objects[label], do_unlink=True)

# Ensure the correct number of command-line arguments

# Unpack command-line arguments
blend_path, input_recons, xml_path = map(str, sys.argv[-3:])

# blend_path = str(sys.argv[-3])
# input_recons = str(sys.argv[-2])
# xml_path = str(sys.argv[-1])
try:
    delete_object('Cube')
    delete_object('Light')
except:
    pass
try:
    # Get origin coordinate from XML file
    with open(xml_path, encoding='utf-8') as file_object:
        all_the_xmlStr = file_object.read()
        dictdata = xmltodict.parse(all_the_xmlStr)
        origin_coord = dictdata['ModelMetadata']['SRSOrigin']

    # Load .obj files into Blender and reset rotation
    for tile in os.listdir(input_recons):
        obj_path = os.path.join(input_recons, tile, f"{tile}.obj")
        if os.path.exists(obj_path):
            bpy.ops.import_scene.obj(filepath=obj_path)
            obj = bpy.context.selected_objects[0]
            obj.rotation_euler = (0, 0, 0)
        else:
            print(f"Warning: Object file not found - {obj_path}")

    # Configure Blender rendering engine and settings
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display.shading.light = 'FLAT'
    bpy.context.scene.display.shading.color_type = 'TEXTURE'
    bpy.context.scene.render.image_settings.file_format = 'JPEG'

    # Save the Blender file
    bpy.ops.wm.save_mainfile(filepath=blend_path, compress=False)
    print(f"Blender file saved to: {blend_path}")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)