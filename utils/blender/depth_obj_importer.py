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

# Unpack command-line arguments
blend_path, input_recons, xml_path = map(str, sys.argv[-3:])
try:
    delete_object('Cube')
    delete_object('Light')
except:
    pass
try:
    # Parse XML file to get the origin
    with open(xml_path, encoding='utf-8') as file_object:
        all_the_xmlStr = file_object.read()
        dictdata = dict(xmltodict.parse(all_the_xmlStr))
        origin_coord = dictdata['ModelMetadata']['SRSOrigin']

    # Load and reset rotation of obj files
    for tile in os.listdir(input_recons):
        obj_path = os.path.join(input_recons, tile, tile + '.obj')
        if os.path.exists(obj_path):
            bpy.ops.import_scene.obj(filepath=obj_path)
            obj = bpy.context.selected_objects[0]
            obj.rotation_euler = (0, 0, 0)
        else:
            print(f"File not found: {obj_path}")

    # Set render engine and shading options
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.render.image_settings.file_format = "OPEN_EXR"
    bpy.context.scene.render.image_settings.exr_codec = 'ZIP'
    bpy.context.scene.render.image_settings.use_zbuffer = True
    bpy.context.scene.render.image_settings.color_depth = '32'
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True  

    # Save the Blender file
    bpy.ops.wm.save_mainfile(filepath=blend_path, compress=False)
    print(f"Blender file saved to: {blend_path}")

except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)