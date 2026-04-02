import bpy
import sys

# get blend project path
file_path = sys.argv[-1]

# create blank project
bpy.ops.wm.read_factory_settings()

# save
bpy.ops.wm.save_as_mainfile(filepath = file_path)