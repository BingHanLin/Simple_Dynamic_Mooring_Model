import pandas as pd
import sys
import os
import shutil
from .enviroment import Ocean
from .utility import auto_connect_objects, connect_objects, runge_kutta4


class Model():

    def __init__(self, param_json_file):

        self.param_json_file = param_json_file

        self.ocean_data = Ocean(param_json_file)

        self.moor_obj_dict = {}

    def add(self, moor_obj):

        if moor_obj.name in self.moor_obj_dict:
            raise ValueError(
                "the name \"{}\" is already in use.".format(moor_obj.name))

        moor_obj.load_property(self.param_json_file)
        moor_obj.init_node_element()

        self.moor_obj_dict[moor_obj.name] = moor_obj

    def connect(self, *moor_obj_name_list):

        # replace objec name by object instance.
        for moor_obj_name in moor_obj_name_list:
            moor_obj_name[0] = self.moor_obj_dict[moor_obj_name[0]]

        # if the node index is not define, then use auto connection.
        if len(moor_obj_name_list[0]) == 3:
            connect_objects(*moor_obj_name_list)
        else:
            auto_connect_objects(*moor_obj_name_list)

    def run(self, dir_name, delta_t, num_time_steps, save_time_steps, display_time_steps=0):

        # Create folder for output data
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        else:
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)

        moor_obj_list = self.moor_obj_dict.values()

        for time_step in range(num_time_steps):

            present_time = (time_step+1)*delta_t

            runge_kutta4(moor_obj_list, present_time, delta_t, self.ocean_data)

            if (display_time_steps != 0):
                if (time_step % display_time_steps == 0):
                    self.display(present_time)

            if (time_step % save_time_steps == 0):
                save_header = 0
                for obj in moor_obj_list:
                    self.save_node_data_csv(
                        obj, present_time, dir_name, save_header)
                    save_header = 1

    # =======================================
    # 儲存結果 node
    # =======================================

    def save_node_data_csv(self, obj, present_time, dir_name, save_header):

        file_name = './' + dir_name + '/' + 'node_data' + \
            str("%.5f" % (present_time))+'.csv'

        datafdict = {}

        title_list = ['name', 'pos_x', 'pos_y', 'pos_z', 'vel_x',
                      'vel_y', 'vel_z', 'force_x', 'force_y', 'force_z']

        datafdict[title_list[0]] = [obj.name]*obj.num_node
        datafdict[title_list[1]] = obj.node_pos[0, :]
        datafdict[title_list[2]] = obj.node_pos[1, :]
        datafdict[title_list[3]] = obj.node_pos[2, :]
        datafdict[title_list[4]] = obj.node_vel[0, :]
        datafdict[title_list[5]] = obj.node_vel[1, :]
        datafdict[title_list[6]] = obj.node_vel[2, :]
        datafdict[title_list[7]] = obj.node_force[0, :]
        datafdict[title_list[8]] = obj.node_force[1, :]
        datafdict[title_list[9]] = obj.node_force[2, :]

        dataframe = pd.DataFrame(datafdict)

        if save_header == 0:
            dataframe.to_csv(file_name, sep=',', mode='a', columns=title_list)
        else:
            dataframe.to_csv(file_name, sep=',', mode='a',
                             header=False, columns=title_list)

    def display(self, present_time):
        print("Time ", present_time)

    def get_object(self, obj_name):
        return self.moor_obj_dict[obj_name]

    def summary(self):
        table_width = 11
        table_width_str = str(table_width)

        print("="*table_width*2)

        print(("{:^"+table_width_str+"s}|{:^"+table_width_str +
               "s}|").format("Name", "N_elements"))

        for _, moor_obj in self.moor_obj_dict.items():
            print(("{:^"+table_width_str+"s}|{:^"+table_width_str +
                   "d}|").format(moor_obj.name, moor_obj.num_element))

        print("="*table_width*2)
