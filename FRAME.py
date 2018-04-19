import json
import math
import numpy as np

from Structures import STRUCTURES

class FRAME(STRUCTURES):
    '''
    Parameters
    ----------
    name : name of the object

    filename : name of parameter file
    
    ocean : wave current 

    start_node : location of start node in 3D (on edge of circle)
    '''

    '''
    c2                             c3
    ################################
    #              #               #
    #              #               #
    #              #               #
    #              #               #
    ################################
    c1               0             c4
    '''

    def __init__(self, name, filename, OCEAN, start_node):

        with open(filename, 'r') as load_f:
            params_dict = json.load(load_f)

        # moment of inertia
        self.global_inertia = np.zeros(3)

        self.global_inertia[0] = params_dict["FRAME"]["MomentInertiaX"]
        self.global_inertia[1] = params_dict["FRAME"]["MomentInertiaY"]
        self.global_inertia[2] = params_dict["FRAME"]["MomentInertiaZ"]

        self.frame_diameter  = params_dict["FRAME"]["TubeDiameter"]
        self.frame_length = params_dict["FRAME"]["Length"]
        self.frame_width = params_dict["FRAME"]["Width"]
        self.density        = params_dict["FRAME"]["MaterialDensity"]
        self.intertia_coeff = params_dict["FRAME"]["InertiaCoefficient"]

        self.element_length    = params_dict["FRAME"]["ElementLength"]
        
        if (self.frame_length % self.element_length != 0 or  self.frame_width % self.element_length != 0):
            raise ValueError("Check element length of frame")


        self.__cal_const_var()
        self.__init_element( np.asarray(start_node) )

        super().__init__(OCEAN, name, 0)


        print("FRAME is built.")

    # =======================================
    # 計算浮框相關常數
    # =======================================
    def __cal_const_var(self):

        # element number in length and width
        self.num_m = self.frame_length / self.element_length
        self.num_n = self.frame_width / self.element_length
        self.num_element = int(self.num_m*2 + self.num_n*3)

        self.num_node = self.num_element - 1
        # total mass
        total_mass = (0.25*self.frame_diameter**2*math.pi) * (self.frame_length*2 + self.frame_width*3) * self.density
	    
        # mass of each element
        self.element_mass = total_mass / self.num_element

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
        temp1 = 1
        temp2 = 1
        temp3 = 1
        temp4 = 1
        temp5 = 1
        for i in range(self.num_node):

            if i < self.num_m/2:
                self.global_node_position[0, i] = ( start_node[0] - self.element_length*(i+1) )
                                    
                self.global_node_position[1, i] = start_node[1]
                
            elif i < self.num_m/2 + self.num_n:

                self.global_node_position[0, i] = ( start_node[0] - self.element_length*self.num_m/2 )
                                    
                self.global_node_position[1, i] = ( start_node[1] + self.element_length*temp1 )
                temp1 = temp1 + 1

            elif i < 3*self.num_m/2 + self.num_n:
                self.global_node_position[0, i] = ( start_node[0] - self.element_length*self.num_m/2
                                                                  + self.element_length*temp2 )
                                    
                self.global_node_position[1, i] = ( start_node[1] + self.element_length*self.num_n )
                temp2 = temp2 + 1

            elif i < 3*self.num_m/2 + 2*self.num_n:
                self.global_node_position[0, i] = ( start_node[0] + self.element_length*self.num_m/2 )
                                    
                self.global_node_position[1, i] = ( start_node[1] + self.element_length*self.num_n
                                                                  - self.element_length*temp3 )
                temp3 = temp3 + 1

            elif i < 4*self.num_m/2 + 2*self.num_n:
                self.global_node_position[0, i] = ( start_node[0] + self.element_length*self.num_m/2
                                                                  - self.element_length*temp4 )
                self.global_node_position[1, i] = ( start_node[1] )
                temp4 = temp4 + 1

            elif i < 4*self.num_m/2 + 3*self.num_n:
                self.global_node_position[0, i] = ( start_node[0] )
                self.global_node_position[1, i] = ( start_node[1] + self.element_length*temp5)
                temp5 = temp5 + 1


            self.global_node_position[2, i] =  start_node[2]

        # 計算角點位置
        self.corner_position = np.zeros((3,4))
        self.corner_index = np.zeros(4, dtype=np.uint8)
    
        self.corner_index[0] = self.num_m/2-1
        self.corner_index[1] = self.num_m/2+self.num_n-1
        self.corner_index[2] = 3*self.num_m/2+self.num_n-1
        self.corner_index[3] = 3*self.num_m/2+2*self.num_n-1
        
        self.corner_position[:,0] = self.global_node_position[:, self.corner_index[0]]
        self.corner_position[:,1] = self.global_node_position[:, self.corner_index[1]]
        self.corner_position[:,2] = self.global_node_position[:, self.corner_index[2]]
        self.corner_position[:,3] = self.global_node_position[:, self.corner_index[3]]

        # global position of center of collar
        self.global_center_position = np.zeros(3)

        self.global_center_position[0] = start_node[0]
        self.global_center_position[1] = start_node[1] + self.element_length*self.num_n/2
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
    # 更新質點位置
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

    # =======================================
    # 計算構件受力
    # =======================================
    def cal_node_force(self,  present_time, global_center_position_temp, combine_center_velocity_temp):
 
        # 初始化流阻力、慣性力、浮力、重力、附加質量
        flow_resistance_force =  np.zeros((3,self.num_element))
        inertial_force =  np.zeros((3,self.num_element))
        buoyancy_force =  np.zeros((3,self.num_element))
        gravity_force =  np.zeros((3,self.num_element))
        connected_force = np.zeros((3,self.num_node))
        added_mass_element = np.zeros(self.num_element)
        self.node_force = np.zeros((3,self.num_node))
        self.pass_force = np.zeros((3,self.num_node))

        local_center_velocity_temp = np.zeros(6)
        local_center_velocity_temp[3:6] = np.copy(combine_center_velocity_temp[3:6])

        # 用 center position 算 global_node_position_temp
        global_center_velocity_temp = np.zeros(6)
        global_center_velocity_temp[0:3] = np.copy(combine_center_velocity_temp[0:3])
        global_center_velocity_temp[3:6] = np.dot(self.coordinate_transfer_matrix.T, local_center_velocity_temp[3:6])


        global_node_position_temp = np.zeros((3,self.num_node))
        global_node_velocity_temp = np.zeros((3,self.num_node))

        for i in range(self.num_node):
            global_node_position_temp[:,i] = np.dot(self.coordinate_transfer_matrix.T, self.local_position_vectors[:,i]) + global_center_position_temp
            global_node_velocity_temp[:,i] = global_center_velocity_temp[0:3] + np.cross(global_center_velocity_temp[3:6], self.local_position_vectors[:, i])

 

        for i in range(self.num_element):            
            node_index = self.get_node_index(i)
            # 構件質心處海波流場
            water_velocity, water_acceleration = self.OCEAN.cal_wave_field(
                                                (global_node_position_temp[0, node_index[0]] + global_node_position_temp[0, node_index[1]])/2,
                                                (global_node_position_temp[1, node_index[0]] + global_node_position_temp[1, node_index[1]])/2,
                                                (global_node_position_temp[2, node_index[0]] + global_node_position_temp[2, node_index[1]])/2,
                                                present_time )

            # 構件與海水相對速度
            relative_velocity  = water_velocity - 0.5*(global_node_velocity_temp[:,node_index[0]] + global_node_velocity_temp[:,node_index[1]])
            relative_velocity_abs = np.linalg.norm(relative_velocity)

            if relative_velocity_abs == 0:
                relative_velocity_unit = np.zeros(3)
            else:
                relative_velocity_unit = relative_velocity / relative_velocity_abs
                
            # 構件切線向量
            element_tang_vecotr =  [ global_node_position_temp[0, node_index[1]] - global_node_position_temp[0, node_index[0]],
                                     global_node_position_temp[1, node_index[1]] - global_node_position_temp[1, node_index[0]],
                                     global_node_position_temp[2, node_index[1]] - global_node_position_temp[2, node_index[0]] ]

            element_length = np.linalg.norm(element_tang_vecotr)
            element_tang_unitvector = element_tang_vecotr / element_length

            # 構件單位切線相對速度
            relative_velocity_unit_tang  = np.dot(element_tang_unitvector,relative_velocity_unit)*element_tang_unitvector

            # 構件單位法線相對速度
            relative_velocity_unit_norm = relative_velocity_unit - relative_velocity_unit_tang

            # 液面至構件質心垂直距離
            dh = self.OCEAN.eta - (global_node_position_temp[2, node_index[1]] + global_node_position_temp[2, node_index[0]])/2

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
            
            flow_resistance_force[:, i] = 0.5*(flow_resistance_force_tang + flow_resistance_force_norm)
            
            # 慣性力
            inertial_force[:, i] = self.OCEAN.water_density*self.intertia_coeff*volume_in_water*water_acceleration*0.5

            if ( np.linalg.norm(water_acceleration) != 0):
                added_mass_element[i] = 0.5*(self.intertia_coeff-1)*self.OCEAN.water_density*volume_in_water
            else:
                added_mass_element[i] = 0

            # 浮力
            buoyancy_force[:, i] = np.asarray([   0,
                                                  0,
                                                  self.OCEAN.water_density*volume_in_water*self.OCEAN.gravity])

            
            # 重力
            gravity_force[:, i] = np.asarray([   0,
                                                 0, 
                                                - self.element_mass*self.OCEAN.gravity])
            
        # 連結力
        for connection in self.connections:
            connected_force[:, connection["self_node"]] += connection["object"].pass_force[:,connection['object_node']]


        moment_global_axes = np.zeros(3)
        total_force = np.zeros(3)
        total_mass = 0


        # 質點總合力
        for i in range(self.num_node):
            
            node_mass = 0
            element_index = self.get_element_index(i)

            for index in element_index:

                self.pass_force[:,i] += (   
                                            flow_resistance_force[:, index]
                                            +  inertial_force[:, index] 
                                            +  buoyancy_force[:, index]                
                                            +  gravity_force[:, index]
                                        )
                
                node_mass += added_mass_element[index]

            self.pass_force[:,i] = self.pass_force[:,i]/len(element_index)

            self.node_force[:,i] = self.pass_force[:,i] + connected_force[:, i]

            node_mass =  node_mass/len(element_index) + (self.element_mass)/2*2

            moment_global_axes = moment_global_axes + np.cross( self.local_position_vectors[:,i], self.node_force[:,i] )

            total_force = total_force + self.node_force[:,i]

            total_mass = total_mass + node_mass

        # ============================================

        moment_local_axes = np.dot(self.coordinate_transfer_matrix, moment_global_axes)


        local_inertia = self.global_inertia

        # Xg, Yg, Zg
        global_center_acc_temp = np.cross( global_center_velocity_temp[0:3], local_center_velocity_temp[3:6]) + total_force/total_mass


        local_center_acc_temp = np.zeros(6)
        # local
        local_center_acc_temp[3] = ( -( local_inertia[2] -local_inertia[1] )*local_center_velocity_temp[4]*local_center_velocity_temp[5]/local_inertia[0]
                                     + moment_local_axes[0]/local_inertia[0] )

        local_center_acc_temp[4] = ( -( local_inertia[0] -local_inertia[2] )*local_center_velocity_temp[3]*local_center_velocity_temp[5]/local_inertia[1]
                                     + moment_local_axes[1]/local_inertia[1] )

        local_center_acc_temp[5] = ( -( local_inertia[1] -local_inertia[0] )*local_center_velocity_temp[3]*local_center_velocity_temp[4]/local_inertia[2]
                                     + moment_local_axes[2]/local_inertia[2] )

        combine_center_acc_temp = np.hstack((global_center_acc_temp[0:3], local_center_acc_temp[3:6]))

        return global_center_velocity_temp[0:3], combine_center_acc_temp


    # =======================================
    # 質點 構件 關係
    # =======================================

    def get_node_index(self, element_number):

        if element_number  == 0:
            node_index = [int(4*self.num_m/2 + 2*self.num_n - 1), 0]
        elif element_number == self.num_element-1:
            node_index = [-1, int(2*self.num_m/2 + self.num_n - 1)]
        else:
            node_index = [element_number-1, element_number]

        return node_index


    def get_element_index(self, node_number):

        if node_number  == 2*self.num_m/2 + self.num_n -1:
            element_index = [node_number, node_number+1, -1]

        elif node_number  == 4*self.num_m/2 + 2*self.num_n -1:
            element_index = [0, node_number, node_number+1]
            
        else:
            element_index = [node_number, node_number+1]

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
