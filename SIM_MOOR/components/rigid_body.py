import numpy as np

from .fram_work import FrameWork

class RigidBody(FrameWork):
    # ==============================================================================
    # Runge Kutta 4th 取得質點位置 速度
    # ==============================================================================
    def get_rk4_position_velocity(self):
        return  self.obj_pos, np.append(self.obj_vel, self.local_obj_omega)
        
    # ==============================================================================
    # Runge Kutta 4th 更新質點位置 速度
    # ==============================================================================
    def update_position_velocity(self):

        self.obj_pos = np.copy(self.rk4_node_pos)
        self.obj_vel = np.copy(self.rk4_node_vel[0:3])
        self.local_obj_omega = np.copy(self.rk4_node_vel[3:6])

        self.node_pos, self.node_vel = self.trans_pos_vel(self.rk4_node_pos, self.rk4_node_vel)

        self.bryant_angle = self.get_bryant_angle(self.local_obj_omega)

    # ==============================================================================
    # Runge Kutta 4th 更新質點位置 速度
    # ==============================================================================
    def get_correct_position_velocity(self):
        return self.trans_pos_vel(self.rk4_node_pos, self.rk4_node_vel)

    # ==============================================================================
    # 轉換求得質點位置及速度
    # ==============================================================================
    def trans_pos_vel(self, obj_pos, combine_obj_vel):

        obj_pos_temp = np.copy(obj_pos)
        obj_vel_temp = np.copy(combine_obj_vel[0:3])
        local_obj_omega_temp = np.copy(combine_obj_vel[3:6])

        bryant_angle_temp = self.get_bryant_angle(local_obj_omega_temp)
        coordinate_transfer_matrix = self.get_coordinate_transform_matrix(bryant_angle_temp)

        obj_omega = np.dot(coordinate_transfer_matrix.T, local_obj_omega_temp)
        
        node_pos_temp = np.zeros((3,self.num_node))
        node_vel_temp = np.zeros((3,self.num_node))

        for i in range(self.num_node):
            node_pos_temp[:,i] = np.dot(coordinate_transfer_matrix.T, self.obj_node_vector[:,i]) + obj_pos_temp
            node_vel_temp[:,i] = obj_vel_temp + np.cross(obj_omega, self.obj_node_vector[:, i])

        return node_pos_temp, node_vel_temp


    # ==============================================================================
    # 修正質點位置、速度
    # ==============================================================================  
    def correct_node_rk4pos_rk4vel(self):
        pass

    # ==============================================================================
    # 計算回傳速度、加速度
    # ==============================================================================
    def get_rk4vel_rk4acc(self):


        obj_vel_temp = np.copy(self.rk4_node_vel[0:3])
        local_obj_omega_temp = np.copy(self.rk4_node_vel[3:6])

        bryant_angle_temp = self.get_bryant_angle(local_obj_omega_temp)
        coordinate_transfer_matrix = self.get_coordinate_transform_matrix(bryant_angle_temp)

        
        # 質點總合力
        self.node_force = np.copy(self.pass_force)

        # 加上連結力
        for connection in self.connections:
            if connection["self_node_condition"] == 1:
                self.node_force[:, connection["self_node"]] += connection["connect_obj"].pass_force[:,connection['connect_obj_node']]

        total_mass = np.sum(self.node_mass)
        total_force = np.sum(self.node_force, axis=1)
        
        moment_global_axes = np.sum( np.cross( self.obj_node_vector.T, self.node_force.T ), axis=0)
        moment_local_axes = np.dot(coordinate_transfer_matrix, moment_global_axes)

        # Xg, Yg, Zg
        obj_acc_temp = np.cross( obj_vel_temp, local_obj_omega_temp) + total_force/total_mass


        local_obj_rotation_acc_temp = np.zeros(3)

        # local
        local_obj_rotation_acc_temp[0] = ( -( self.moment_inertia[2] -self.moment_inertia[1] )*local_obj_omega_temp[1]*local_obj_omega_temp[2]/self.moment_inertia[0]
                                           + moment_local_axes[0]/self.moment_inertia[0] )

        local_obj_rotation_acc_temp[1] = ( -( self.moment_inertia[0] -self.moment_inertia[2] )*local_obj_omega_temp[0]*local_obj_omega_temp[2]/self.moment_inertia[1]
                                           + moment_local_axes[1]/self.moment_inertia[1] )

        local_obj_rotation_acc_temp[2] = ( -( self.moment_inertia[1] -self.moment_inertia[0] )*local_obj_omega_temp[0]*local_obj_omega_temp[1]/self.moment_inertia[2]
                                           + moment_local_axes[2]/self.moment_inertia[2] )

        combine_center_acc_temp = np.append(obj_acc_temp, local_obj_rotation_acc_temp)


        return obj_vel_temp, combine_center_acc_temp


    # ==============================================================================
    # get byant angle at current time step
    # ==============================================================================
    def get_bryant_angle(self, local_obj_omega_temp):
        
        bryant_angle_temp = np.zeros(3)
        
        bryant_angle_temp[0] =  self.bryant_angle[0] + self.time_marching_dt*((  local_obj_omega_temp[0]*np.cos( self.bryant_angle[2] )
                                        - local_obj_omega_temp[1]*np.sin( self.bryant_angle[2] ) ) / np.cos(self.bryant_angle[1]))    
        
        bryant_angle_temp[1] =  self.bryant_angle[1] + self.time_marching_dt*((  local_obj_omega_temp[0]*np.sin( self.bryant_angle[2] )
                                        + local_obj_omega_temp[1]*np.cos( self.bryant_angle[2] ) )) 

        bryant_angle_temp[2] =  self.bryant_angle[2] + self.time_marching_dt*( local_obj_omega_temp[2] - 
                                       (  local_obj_omega_temp[0]*np.cos( self.bryant_angle[2] )
                                       - local_obj_omega_temp[1]*np.sin( self.bryant_angle[2] ) ) * np.tan(self.bryant_angle[1])) 

        return bryant_angle_temp

    # ==============================================================================
    # 座標轉換
    # ==============================================================================
    def get_coordinate_transform_matrix(self, bryant_angle_temp):
        
        coordinate_transfer_matrix = np.zeros((3,3))

        coordinate_transfer_matrix[0,0] = np.cos(bryant_angle_temp[2])*np.cos(bryant_angle_temp[1])

        coordinate_transfer_matrix[0,1] = (  np.sin(bryant_angle_temp[2])*np.cos(bryant_angle_temp[0])
                                           + np.cos(bryant_angle_temp[2])*np.sin(bryant_angle_temp[1])
                                           * np.sin(bryant_angle_temp[0]) )

        coordinate_transfer_matrix[0,2] = (  np.sin(bryant_angle_temp[2])*np.sin(bryant_angle_temp[0])
                                           - np.cos(bryant_angle_temp[2])*np.sin(bryant_angle_temp[1])
                                           * np.cos(bryant_angle_temp[0]) )

        coordinate_transfer_matrix[1,0] = -np.sin(bryant_angle_temp[2])*np.cos(bryant_angle_temp[1])

        coordinate_transfer_matrix[1,1] = (  np.cos(bryant_angle_temp[2])*np.cos(bryant_angle_temp[0])
                                           - np.sin(bryant_angle_temp[2])*np.sin(bryant_angle_temp[1])
                                           * np.sin(bryant_angle_temp[0]) )

        coordinate_transfer_matrix[1,2] = (  np.cos(bryant_angle_temp[2])*np.sin(bryant_angle_temp[0])
                                           + np.sin(bryant_angle_temp[2])*np.sin(bryant_angle_temp[1])
                                           * np.cos(bryant_angle_temp[0]) )

        coordinate_transfer_matrix[2,0] = np.sin(bryant_angle_temp[1])

        coordinate_transfer_matrix[2,1] = -np.cos(bryant_angle_temp[1])*np.sin(bryant_angle_temp[0])

        coordinate_transfer_matrix[2,2] = np.cos(bryant_angle_temp[1])*np.cos(bryant_angle_temp[0])

        return coordinate_transfer_matrix