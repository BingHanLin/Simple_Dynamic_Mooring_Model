import json
import math
import numpy as np

from Structures import STRUCTURES


class ANCHOR(STRUCTURES):
    '''
    Parameters
    ----------
    name : name of the object

    filename : name of parameter file
    
    ocean : wave current 

    start_node : location of start node in 3D

    end_node : location of end node in 3D
    '''
    
    def __init__(self, name, OCEAN, start_node):

        self.__cal_const_var()
        self.__init_element(np.asarray(start_node))

        super().__init__(OCEAN, name, 1)

        print("Anchor instance is built.")

    # =======================================
    # 計算繫纜相關常數
    # =======================================
    def __cal_const_var(self):

        self.num_node = 1
        self.num_element = 1
    # =======================================
    # 計算初始浮框構件(兩端)質點位置、速度
    # =======================================
    def __init_element(self, start_node):

        # global position, velocity, force of nodes
        self.global_node_position =  np.zeros((3, self.num_node)) 
        self.global_node_velocity = np.zeros((3, self.num_node))
        self.pass_force = np.zeros((3,self.num_node))
        self.global_node_position[:,0] = start_node

    # =======================================
    # Runge Kutta 4th 更新質點位置
    # =======================================

    def update_position_velocity(self, dt):
        
        for connection in self.connections:
            if connection["connect_obj_node_condition"] == 1:
                self.new_rk4_position[:,connection["self_node"]] = connection["connect_obj"].global_node_position[:,connection['connect_obj_node']]
                self.new_rk4_velocity[:,connection["self_node"]] = connection["connect_obj"].global_node_velocity[:,connection['connect_obj_node']]
            
            elif connection["self_node_condition"] == 1:
                connection["connect_obj"].global_node_position[:,connection['connect_obj_node']] = self.new_rk4_position[:,connection["self_node"]]
                connection["connect_obj"].global_node_velocity[:,connection['connect_obj_node']] = self.new_rk4_velocity[:,connection["self_node"]]

        self.global_node_position = np.copy(self.new_rk4_position)
        self.global_node_velocity = np.copy(self.new_rk4_velocity)

    # =======================================
    # 計算質點位置、速度
    # =======================================  
    def cal_node_pos_vel(self, global_node_position_temp, global_node_velocity_temp):
        
        self.global_node_position = global_node_position_temp
        self.global_node_velocity = global_node_velocity_temp
        
        for connection in self.connections:
            if connection["connect_obj_node_condition"] == 1:
                self.global_node_position[:,connection["self_node"]] = connection["connect_obj"].global_node_position[:,connection['connect_obj_node']]
                self.global_node_velocity[:,connection["self_node"]] = connection["connect_obj"].global_node_velocity[:,connection['connect_obj_node']]
            
            elif connection["self_node_condition"] == 1:
                connection["connect_obj"].global_node_position[:,connection['connect_obj_node']] = self.global_node_position[:,connection["self_node"]]
                connection["connect_obj"].global_node_velocity[:,connection['connect_obj_node']] = self.global_node_velocity[:,connection["self_node"]]

    # =======================================
    # 計算構件受力
    # =======================================
    def cal_element_force(self, present_time):
        pass


    # =======================================
    # 計算回傳速度、加速度
    # =======================================
    def cal_vel_acc(self):

        connected_force = np.zeros((3,self.num_node))

        # 連結力
        for connection in self.connections:
            if connection["self_node_condition"] == 1:
                connected_force[:, connection["self_node"]] += connection["connect_obj"].pass_force[:,connection['connect_obj_node']]

        self.node_force  = connected_force

        global_node_acc_temp = np.zeros((3,self.num_node))
        global_node_velocity_temp = np.zeros((3,self.num_node))

        return global_node_velocity_temp, global_node_acc_temp


    # =======================================
    # 質點 構件 關係
    # =======================================

    def get_node_index(self, element_number):


        node_index = [element_number]

        return node_index

    def get_element_index(self, node_number):

        element_index = [node_number]

        return element_index
        

if __name__ == "__main__":
     pass
