import os
import shutil


def create_folder(dir_name, STRUCTURES):

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)

    with open(dir_name+'/_info', 'w+') as out_file:
        
        for obj in STRUCTURES.getinstances():
            out_file.write(obj.name +'\n')


