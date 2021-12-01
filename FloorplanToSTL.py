from utils.FloorplanToBlenderLib import *
from subprocess import check_output
import os


def createFloorPlan(image_path = config.image_path, target_path = config.target_path, SR_Check=True):
    import config 
    SR= [config.SR_scale,config.SR_method]
    program_path = config.program_path
    blender_install_path = config.blender_install_path
    blender_script_path = config.blender_script_path
    CubiCasa = config.CubiCasa
    data_paths = [execution.simple_single(image_path,CubiCasa,SR)]
  
    check_output([blender_install_path,
     "-noaudio", # this is a dockerfile ubuntu hax fix
     "--background",
     "--python",
     blender_script_path,
     program_path, # Send this as parameter to script
     target_path
     ] +  data_paths)
    
    print("Created File at "+target_path)
