import numpy as np

from .fram_work import FrameWork

class NonRigidBody(FrameWork):

    # ==============================================================================
    # Runge Kutta 4th 取得質點位置 速度
    # ==============================================================================
    def get_rk4_position_velocity(self):
        return np.copy(self.node_pos), np.copy(self.node_vel)

    # ==============================================================================
    # Runge Kutta 4th 更新質點位置 速度
    # ==============================================================================
    def update_position_velocity(self):
        self.node_pos = np.copy(self.rk4_node_pos)
        self.node_vel = np.copy(self.rk4_node_vel)

    # ==============================================================================
    # Runge Kutta 4th 求得質點位置及速度
    # ==============================================================================
    def get_correct_position_velocity(self):
        return np.copy(self.rk4_node_pos), np.copy(self.rk4_node_vel)

    # ==============================================================================
    # 修正質點位置、速度
    # ==============================================================================  
    def correct_node_rk4pos_rk4vel(self):
        for connection in self.connections:
            if connection["connect_obj_node_condition"] == 1:
                self.rk4_node_pos[:,connection["self_node"]] = connection["connect_obj"].correct_node_pos[:,connection['connect_obj_node']]
                self.rk4_node_vel[:,connection["self_node"]] = connection["connect_obj"].correct_node_vel[:,connection['connect_obj_node']]

    # ==============================================================================
    # 計算回傳速度、加速度
    # ==============================================================================
    def get_rk4vel_rk4acc(self):

        # 質點總合力
        self.node_force = np.copy(self.pass_force)

        # 加上連結力
        for connection in self.connections:
            if connection["self_node_condition"] == 1:
                self.node_force[:, connection["self_node"]] += connection["connect_obj"].pass_force[:,connection['connect_obj_node']]
                self.node_mass[connection["self_node"]] += connection["connect_obj"].node_mass[connection['connect_obj_node']]

        # 加速度
        node_acc = np.where(self.node_mass == 0, 0,  self.node_force / self.node_mass)


        return self.rk4_node_vel, node_acc

