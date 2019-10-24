import json
import math
import numpy as np
from .non_rigid_body import NonRigidBody


class Anchor(NonRigidBody):
    '''
    Parameters
    ----------
    name : name of the object

    start_node : location of start node in 3D
    '''
    
    # ==============================================================================
    # 引入材質參數
    # ==============================================================================
    def load_property(self, param_json_file):
        pass

    # ==============================================================================
    # 初始化構件與質點
    # ==============================================================================
    def init_node_element(self):

        self.num_node = 1
        self.num_element = 0

        # global position, velocity, force of nodes
        self.node_pos =  np.zeros((3, self.num_node)) 
        self.node_vel = np.zeros((3, self.num_node))
        self.pass_force = np.zeros((3,self.num_node))
        self.node_force = np.zeros((3,self.num_node))
        self.node_pos[:,0] = self.start_node_pos
        self.node_mass = np.zeros(self.num_node)

    # ==============================================================================
    # 計算構件受力
    # ==============================================================================
    def cal_element_force(self, node_pos_temp, node_vel_temp, present_time, ocean_data):
        pass

    # ==============================================================================
    # 計算回傳速度、加速度(Anchor 用)
    # ==============================================================================
    def get_rk4vel_rk4acc(self):

        # 質點總合力
        self.node_force = np.copy(self.pass_force)

        # 加速度 (強制為0)
        node_acc = np.where(self.node_mass == 0, 0, 0)

        node_vel_temp = np.zeros_like(self.rk4_node_vel)

        return  node_vel_temp, node_acc

    # ==============================================================================
    # 質點 構件 關係
    # ==============================================================================
    def get_node_index(self, element_number):
        node_index = [element_number]

        return node_index

    def get_element_index(self, node_number):
        element_index = [node_number]

        return element_index
        

if __name__ == "__main__":
     pass
