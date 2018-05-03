import json
import math
import numpy as np

from Structures import STRUCTURES

class COLLAR(STRUCTURES):
    '''
    Parameters
    ----------
    name : name of the object

    filename : name of parameter file
    
    ocean : wave current 

    start_node : location of start node in 3D (on edge of circle)
    '''
    
    def __init__(self, name, filename, OCEAN, start_node):

        with open(filename, 'r') as load_f:
            params_dict = json.load(load_f)

        self.float_type = params_dict["COLLAR"]["FloatingType"]

        self.circum_out     = params_dict["COLLAR"]["OutterCircumference"]
        self.circum_in      = params_dict["COLLAR"]["InnerCircumference"]
        self.tube_diameter  = params_dict["COLLAR"]["TubeDiameter"]
        self.tube_thickness = params_dict["COLLAR"]["TubeThickness"]
        
        self.density        = params_dict["COLLAR"]["MaterialDensity"]
        self.intertia_coeff = params_dict["COLLAR"]["InertiaCoefficient"]

        self.num_element    = params_dict["COLLAR"]["Elements"]
        self.other_mass     = params_dict["COLLAR"]["OtherMass"]


        self.__cal_const_var()
        self.__init_element( np.asarray(start_node) )

        super().__init__(OCEAN, name, 0)


        print("Collar is built.")

    # =======================================
    # 計算浮框相關常數
    # =======================================
    def __cal_const_var(self):

        self.num_node = self.num_element

        # radius of the collar
        self.collar_radius_out = self.circum_out / 2 / math.pi
        self.collar_radius_in  = self.circum_out / 2 / math.pi

        # total mass
        if (self.float_type == 0):  # floating
            total_mass = (0.25 * math.pi*
                (self.tube_diameter**2 - (self.tube_diameter - self.tube_thickness * 2)**2)*
                (self.circum_out + self.circum_in) * self.density + self.other_mass)

        elif (self.float_type == 1):  # sinking 
            total_mass = (0.25 * math.pi * 
                (self.tube_diameter**2 - (self.tube_diameter - self.tube_thickness * 2)**2) *
                (self.circum_out + self.circum_in) * self.density + self.OCEAN.water_density *
                (0.25 * math.pi * (self.tube_diameter - self.tube_thickness * 2)**2 *
                (self.circum_out + self.circum_in)))


        # moment of inertia
        self.global_inertia = np.zeros(3)

        self.global_inertia[0] = total_mass*self.collar_radius_in**2/2
        self.global_inertia[1] = total_mass*self.collar_radius_in**2/2
        self.global_inertia[2] = total_mass*self.collar_radius_in**2
	    
        # mass of each element
        self.element_mass = total_mass / self.num_element
        # angle of each element
        self.theta_element = 2*np.pi/ self.num_element

        # coordinate transfer
        self.bryant_angle_phi = np.zeros(3)

        # create coordinate transformation matrix
        self.coordinate_transfer_matrix = np.zeros((3,3))

    # =======================================
    # 計算初始浮框構件(兩端)質點位置、速度等
    # =======================================
    def __init_element(self, start_node):

        # global position and velocity of nodes
        self.global_node_position = np.zeros((3,self.num_node))
        self.global_node_velocity = np.zeros((3,self.num_node))
        self.pass_force = np.zeros((3,self.num_node))

        for i in range(self.num_node):
            self.global_node_position[0, i] = (  start_node[0]
                                   + self.collar_radius_in*np.cos(self.theta_element*i-np.pi)
                                   + self.collar_radius_in )
            self.global_node_position[1, i] =  start_node[1] - self.collar_radius_in*np.sin(self.theta_element*i-np.pi)
            self.global_node_position[2, i] =  start_node[2]

        # global position of center of collar
        self.global_center_position = np.zeros(3)

        self.global_center_position[0] = start_node[0] + self.collar_radius_in
        self.global_center_position[1] = start_node[1]
        self.global_center_position[2] = start_node[2]  

        self.global_center_velocity = np.zeros(6)
        self.local_center_velocity = np.zeros(6)  # 0~2 always equal to 0

        # local position vectors of nodes
        self.local_position_vectors = np.zeros((3,self.num_node))

        for i in range(self.num_node):
            self.local_position_vectors[0, i] = self.global_node_position[0, i] - self.global_center_position[0]
            self.local_position_vectors[1, i] = self.global_node_position[1, i] - self.global_center_position[1]
            self.local_position_vectors[2, i] = self.global_node_position[2, i] - self.global_center_position[2]
            
    # =======================================
    # Runge Kutta 4th 更新質點位置
    # =======================================
    def update_position_velocity(self, dt):

        self.global_center_position = np.copy(self.new_rk4_position)

        self.global_center_velocity[0:3] = np.copy(self.new_rk4_velocity[0:3])
        
        self.local_center_velocity[3:6] = np.copy(self.new_rk4_velocity[3:6])


        # 用 center position 算 global_node_position_temp
        self.global_center_velocity[3:6] = np.dot(self.coordinate_transfer_matrix.T, self.local_center_velocity[3:6])

        for i in range(self.num_node):
            self.global_node_position[:,i] = np.dot(self.coordinate_transfer_matrix.T, self.local_position_vectors[:,i]) + self.global_center_position
            self.global_node_velocity[:,i] = self.global_center_velocity[0:3] + np.cross(self.global_center_velocity[3:6], self.local_position_vectors[:, i])
        
        for connection in self.connections:
            if connection["connect_obj_node_condition"] == 1:
                self.global_node_position[:,connection["self_node"]] = connection["connect_obj"].global_node_position[:,connection['connect_obj_node']]
                self.global_node_velocity[:,connection["self_node"]] = connection["connect_obj"].global_node_velocity[:,connection['connect_obj_node']]
            
            elif connection["self_node_condition"] == 1:
                connection["connect_obj"].global_node_position[:,connection['connect_obj_node']] = self.global_node_position[:,connection["self_node"]]
                connection["connect_obj"].global_node_velocity[:,connection['connect_obj_node']] = self.global_node_velocity[:,connection["self_node"]]
    
    # =======================================
    # 計算質點位置、速度
    # =======================================  
    def cal_node_pos_vel(self, global_center_position_temp, combine_center_velocity_temp):
        
        self.global_center_position = np.copy(global_center_position_temp)

        self.local_center_velocity = np.zeros(6)
        self.local_center_velocity[3:6] = np.copy(combine_center_velocity_temp[3:6])

        # 用 center position 算 global_node_position_temp
        self.global_center_velocity = np.zeros(6)
        self.global_center_velocity[0:3] = np.copy(combine_center_velocity_temp[0:3])
        self.global_center_velocity[3:6] = np.dot(self.coordinate_transfer_matrix.T, self.local_center_velocity[3:6])


        self.global_node_position = np.zeros((3,self.num_node))
        self.global_node_velocity = np.zeros((3,self.num_node))

        for i in range(self.num_node):
            self.global_node_position[:,i] = np.dot(self.coordinate_transfer_matrix.T, self.local_position_vectors[:,i]) + self.global_center_position
            self.global_node_velocity[:,i] = self.global_center_velocity[0:3] + np.cross(self.global_center_velocity[3:6], self.local_position_vectors[:, i])

 

    # =======================================
    # 計算構件受力
    # =======================================
    def cal_element_force(self,  present_time):
 
        # 初始化流阻力、慣性力、浮力、重力、附加質量
        self.flow_resistance_force =  np.zeros((3,self.num_element))
        self.inertial_force =  np.zeros((3,self.num_element))
        self.buoyancy_force =  np.zeros((3,self.num_element))
        self.gravity_force =  np.zeros((3,self.num_element))
        self.added_mass_element = np.zeros(self.num_element)
        self.pass_force = np.zeros((3,self.num_node))


        for i in range(self.num_element):            
            node_index = self.get_node_index(i)
            # 構件質心處海波流場
            water_velocity, water_acceleration = self.OCEAN.cal_wave_field(
                                                (self.global_node_position[0, node_index[0]] + self.global_node_position[0, node_index[1]])/2,
                                                (self.global_node_position[1, node_index[0]] + self.global_node_position[1, node_index[1]])/2,
                                                (self.global_node_position[2, node_index[0]] + self.global_node_position[2, node_index[1]])/2,
                                                present_time )

            # 構件與海水相對速度
            relative_velocity  = water_velocity - 0.5*(self.global_node_velocity[:,node_index[0]] + self.global_node_velocity[:,node_index[1]])
            relative_velocity_abs = np.linalg.norm(relative_velocity)

            if relative_velocity_abs == 0:
                relative_velocity_unit = np.zeros(3)
            else:
                relative_velocity_unit = relative_velocity / relative_velocity_abs
                
            # 構件切線向量
            element_tang_vecotr =  [ self.global_node_position[0, node_index[1]] - self.global_node_position[0, node_index[0]],
                                     self.global_node_position[1, node_index[1]] - self.global_node_position[1, node_index[0]],
                                     self.global_node_position[2, node_index[1]] - self.global_node_position[2, node_index[0]] ]

            element_length = np.linalg.norm(element_tang_vecotr)
            element_tang_unitvector = element_tang_vecotr / element_length

            # 構件單位切線相對速度
            relative_velocity_unit_tang  = np.dot(element_tang_unitvector,relative_velocity_unit)*element_tang_unitvector

            # 構件單位法線相對速度
            relative_velocity_unit_norm = relative_velocity_unit - relative_velocity_unit_tang

            # 液面至構件質心垂直距離
            dh = self.OCEAN.eta - (self.global_node_position[2, node_index[1]] + self.global_node_position[2, node_index[0]])/2

	        # 計算浸水深度 面積 表面積
            if ( dh > self.frame_diameter/2 ):
                h = self.frame_diameter
                area = math.pi*self.frame_diameter**2/4
                s = math.pi*self.frame_diameter

            elif (dh > 0):
                temp_theta = np.arcsin(dh/(self.frame_diameter/2))
                h = self.frame_diameter/2+dh
                area = math.pi*self.frame_diameter**2/8+dh*self.frame_diameter*np.cos(temp_theta)/2+self.frame_diameter**2*temp_theta/4
                s = math.pi*self.frame_diameter/2+self.frame_diameter*temp_theta

            elif (dh > -self.frame_diameter/2):
                temp_theta = np.arcsin(abs(dh)/(self.frame_diameter/2))
                h = self.frame_diameter/2+dh
                area = math.pi*self.frame_diameter**2/8-dh*self.frame_diameter*np.cos(temp_theta)/2-self.frame_diameter**2*temp_theta/4
                s = math.pi*self.frame_diameter/2-self.frame_diameter*temp_theta  

            else:
                h = 0
                area = 0
                s = 0

            # 浸沒體積
            volume_in_water = math.pi*self.frame_diameter**2*element_length*2/4 # 有內框以及外框
            # 流阻力受力面積
            area_norm = self.frame_diameter*element_length*2      
            area_tang = math.pi*self.frame_diameter*element_length*2

            # 計算流阻力係數  依blevins經驗公式
            afa = np.arccos( np.dot(relative_velocity_unit, relative_velocity_unit_tang) )
            cd_norm = 1.2*(np.sin(afa))**2
            cd_tang = 0.083*np.cos(afa)-0.035*(np.cos(afa))**2

            if(afa > math.pi/2 ):  #??
                afa = math.pi - afa
                element_tang_unitvector = -element_tang_unitvector

            # 流阻力 (計算切線及法線方向流阻力分量)
            flow_resistance_force_tang = 0.5*self.OCEAN.water_density*cd_tang*area_tang*relative_velocity_abs**2*relative_velocity_unit_tang

            flow_resistance_force_norm = 0.5*self.OCEAN.water_density*cd_norm*area_norm*relative_velocity_abs**2*relative_velocity_unit_norm
            
            self.flow_resistance_force[:, i] = 0.5*(flow_resistance_force_tang + flow_resistance_force_norm)
            
            # 慣性力
            self.inertial_force[:, i] = self.OCEAN.water_density*self.intertia_coeff*volume_in_water*water_acceleration*0.5

            if ( np.linalg.norm(water_acceleration) != 0):
                self.added_mass_element[i] = 0.5*(self.intertia_coeff-1)*self.OCEAN.water_density*volume_in_water
            else:
                self.added_mass_element[i] = 0

            # 浮力
            self.buoyancy_force[:, i] = np.asarray([ 0,
                                                     0,
                                                     self.OCEAN.water_density*volume_in_water*self.OCEAN.gravity])

            
            # 重力
            self.gravity_force[:, i] = np.asarray([ 0,
                                                    0, 
                                                    - self.element_mass*self.OCEAN.gravity])
            
        # 質點總合力
        for i in range(self.num_node):
            
            node_mass = 0
            element_index = self.get_element_index(i)

            for index in element_index:

                self.pass_force[:,i] += (   
                                               self.flow_resistance_force[:, index]
                                            +  self.inertial_force[:, index] 
                                            +  self.buoyancy_force[:, index]                
                                            +  self.gravity_force[:, index]
                                        )
    # =======================================
    # 計算回傳速度、加速度
    # =======================================
    def cal_vel_acc(self):

        connected_force = np.zeros((3,self.num_node))
        self.node_force = np.zeros((3,self.num_node))

        # 連結力
        for connection in self.connections:
            if connection["self_node_condition"] == 1:
                connected_force[:, connection["self_node"]] += connection["connect_obj"].pass_force[:,connection['connect_obj_node']]


        moment_global_axes = np.zeros(3)
        total_force = np.zeros(3)
        total_mass = 0


        # 質點總合力
        for i in range(self.num_node):
            
            node_mass = 0
            element_index = self.get_element_index(i)

            for index in element_index:

                self.node_force[:,i] += (   
                                               self.flow_resistance_force[:, index]
                                            +  self.inertial_force[:, index] 
                                            +  self.buoyancy_force[:, index]                
                                            +  self.gravity_force[:, index]
                                        )
                
                node_mass += self.added_mass_element[index]

            self.node_force[:,i] = self.node_force[:,i]/len(element_index) + connected_force[:, i]


            node_mass =  node_mass/len(element_index) + (self.element_mass)/2*2

            moment_global_axes = moment_global_axes + np.cross( self.local_position_vectors[:,i], self.node_force[:,i] )

            total_force = total_force + self.node_force[:,i]

            total_mass = total_mass + node_mass

        # ============================================

        moment_local_axes = np.dot(self.coordinate_transfer_matrix, moment_global_axes)


        local_inertia = self.global_inertia

        # Xg, Yg, Zg
        global_center_acc_temp = np.cross( self.global_center_velocity[0:3], self.local_center_velocity[3:6]) + total_force/total_mass


        local_center_acc_temp = np.zeros(6)
        # local
        local_center_acc_temp[3] = ( -( local_inertia[2] -local_inertia[1] )*self.local_center_velocity[4]*self.local_center_velocity[5]/local_inertia[0]
                                     + moment_local_axes[0]/local_inertia[0] )

        local_center_acc_temp[4] = ( -( local_inertia[0] -local_inertia[2] )*self.local_center_velocity[3]*self.local_center_velocity[5]/local_inertia[1]
                                     + moment_local_axes[1]/local_inertia[1] )

        local_center_acc_temp[5] = ( -( local_inertia[1] -local_inertia[0] )*self.local_center_velocity[3]*self.local_center_velocity[4]/local_inertia[2]
                                     + moment_local_axes[2]/local_inertia[2] )

        combine_center_acc_temp = np.hstack((global_center_acc_temp[0:3], local_center_acc_temp[3:6]))


        global_center_velocity_temp = np.copy(self.global_center_velocity)

        return global_center_velocity_temp[0:3], combine_center_acc_temp


    # =======================================
    # 質點 構件 關係
    # =======================================

    def get_node_index(self, element_number):

        if element_number  == self.num_element-1:
            node_index = [-1, 0]
        else:
            node_index = [element_number, element_number+1]

        return node_index


    def get_element_index(self, node_number):

        if node_number  == 0:
            element_index = [-1, 0]
        else:
            element_index = [node_number-1, node_number]

        return element_index

    # =======================================
    # 座標轉換
    # =======================================
    def coordinate_transform(self, dt):
    
        bryant_angle_phi_temp = np.copy( self.bryant_angle_phi )
        

        self.bryant_angle_phi[0] =  bryant_angle_phi_temp[0] + dt*((  self.local_center_velocity[3]*np.cos( bryant_angle_phi_temp[2] )
                                        - self.local_center_velocity[4]*np.sin( bryant_angle_phi_temp[2] ) ) / np.cos(bryant_angle_phi_temp[1]))    
        
        self.bryant_angle_phi[1] =  bryant_angle_phi_temp[1] + dt*((  self.local_center_velocity[3]*np.sin( bryant_angle_phi_temp[2] )
                                        + self.local_center_velocity[4]*np.cos( bryant_angle_phi_temp[2] ) )) 

        self.bryant_angle_phi[2] =  bryant_angle_phi_temp[2] + dt*( self.local_center_velocity[5] - 
                                       (  self.local_center_velocity[3]*np.cos( bryant_angle_phi_temp[2] )
                                       - self.local_center_velocity[4]*np.sin( bryant_angle_phi_temp[2] ) ) * np.tan(bryant_angle_phi_temp[1])) 




        self.coordinate_transfer_matrix[0,0] = np.cos(self.bryant_angle_phi[2])*np.cos(self.bryant_angle_phi[1])

        self.coordinate_transfer_matrix[0,1] = (  np.sin(self.bryant_angle_phi[2])*np.cos(self.bryant_angle_phi[0])
                                           + np.cos(self.bryant_angle_phi[2])*np.sin(self.bryant_angle_phi[1])
                                           * np.sin(self.bryant_angle_phi[0]) )

        self.coordinate_transfer_matrix[0,2] = (  np.sin(self.bryant_angle_phi[2])*np.sin(self.bryant_angle_phi[0])
                                           - np.cos(self.bryant_angle_phi[2])*np.sin(self.bryant_angle_phi[1])
                                           * np.cos(self.bryant_angle_phi[0]) )

        self.coordinate_transfer_matrix[1,0] = -np.sin(self.bryant_angle_phi[2])*np.cos(self.bryant_angle_phi[1])

        self.coordinate_transfer_matrix[1,1] = (  np.cos(self.bryant_angle_phi[2])*np.cos(self.bryant_angle_phi[0])
                                           - np.sin(self.bryant_angle_phi[2])*np.sin(self.bryant_angle_phi[1])
                                           * np.sin(self.bryant_angle_phi[0]) )

        self.coordinate_transfer_matrix[1,2] = (  np.cos(self.bryant_angle_phi[2])*np.sin(self.bryant_angle_phi[0])
                                           + np.sin(self.bryant_angle_phi[2])*np.sin(self.bryant_angle_phi[1])
                                           * np.cos(self.bryant_angle_phi[0]) )

        self.coordinate_transfer_matrix[2,0] = np.sin(self.bryant_angle_phi[1])

        self.coordinate_transfer_matrix[2,1] = -np.cos(self.bryant_angle_phi[1])*np.sin(self.bryant_angle_phi[0])

        self.coordinate_transfer_matrix[2,2] = np.cos(self.bryant_angle_phi[1])*np.cos(self.bryant_angle_phi[0])


if __name__ == "__main__":
    pass
