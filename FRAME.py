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

        self.element_diameter  = params_dict["FRAME"]["TubeDiameter"]
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
        total_mass = (0.25*self.element_diameter**2*math.pi) * (self.frame_length*2 + self.frame_width*3) * self.density
	    
        # mass of each element
        self.element_mass = total_mass / self.num_element

        # coordinate transfer
        self.bryant_angle_phi = np.zeros(3)

        # create coordinate transformation matrix
        self.coordinate_transfer_matrix = np.zeros((3,3))

        self.func = np.vectorize(self.cal_element_under_water)
    # =======================================
    # 計算初始浮框構件(兩端)質點位置、速度等
    # =======================================
    def __init_element(self, start_node):

        # global position and velocity of nodes
        self.global_node_position = np.zeros((3,self.num_node))
        self.global_node_velocity = np.zeros((3,self.num_node))
        self.node_mass = np.zeros(self.num_node)
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

        # 初始化外傳力
        self.pass_force = np.zeros((3,self.num_node))
        # 初始化浮力、重力
        self.buoyancy_force =  np.zeros((3,self.num_element))
        self.gravity_force =  np.zeros((3,self.num_element))

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

    def cal_element_under_water(self,  delta_h):
        # 計算浸水深度 面積 表面積
        if ( delta_h > self.element_diameter/2 ):
            under_water_height = self.element_diameter
            section_area = math.pi*self.element_diameter**2/4
            skin_surface = math.pi*self.element_diameter

        elif (delta_h > 0):
            temp_theta = math.asin(delta_h/(self.element_diameter/2))
            under_water_height = self.element_diameter/2+delta_h
            section_area = math.pi*self.element_diameter**2/8+delta_h*self.element_diameter*math.cos(temp_theta)/2+self.element_diameter**2*temp_theta/4
            skin_surface = math.pi*self.element_diameter/2+self.element_diameter*temp_theta

        elif (delta_h > -self.element_diameter/2):
            temp_theta = math.asin(abs(delta_h)/(self.element_diameter/2))
            under_water_height = self.element_diameter/2+delta_h
            section_area = math.pi*self.element_diameter**2/8-delta_h*self.element_diameter*math.cos(temp_theta)/2-self.element_diameter**2*temp_theta/4
            skin_surface = math.pi*self.element_diameter/2-self.element_diameter*temp_theta  

        else:
            under_water_height = 0
            section_area = 0
            skin_surface = 0

        return under_water_height, section_area, skin_surface
    # =======================================
    # 計算構件受力
    # =======================================
    def cal_element_force(self,  present_time):
 
        # 迭代構件
        node_index = np.asarray(list( map(self.get_node_index, range(self.num_element))))

        center_position = 0.5*( self.global_node_position[:,node_index[:,0]] + self.global_node_position[:,node_index[:,1]] )

        water_velocity, water_acceleration = self.OCEAN.cal_wave_field(center_position, present_time)

        relative_velocity = water_velocity - 0.5*( self.global_node_velocity[:,node_index[:,0]] + self.global_node_velocity[:,node_index[:,1]] )

        # 構件切線向量
        element_tang_vecotr = self.global_node_position[:,node_index[:,1]] - self.global_node_position[:,node_index[:,0]]

        relative_velocity_abs = np.sum(np.abs(relative_velocity)**2,axis=0)**(1./2)
        relative_velocity_unit = np.where(relative_velocity_abs != 0, relative_velocity / relative_velocity_abs, 0) 

        element_length = np.sum(np.abs(element_tang_vecotr)**2,axis=0)**(1./2)
        element_tang_unitvector  = np.where(element_length != 0, element_tang_vecotr / element_length, 0) 

        # 構件單位切線相對速度
        relative_velocity_tang  = np.sum(element_tang_unitvector*relative_velocity,axis=0)*element_tang_unitvector
        relative_velocity_tang_abs = np.sum(np.abs(relative_velocity_tang)**2,axis=0)**(1./2)
        relative_velocity_unit_tang = np.where(relative_velocity_tang_abs != 0, relative_velocity_tang / relative_velocity_tang_abs, 0) 
        
        # 構件單位法線相對速度
        relative_velocity_norm  = relative_velocity - relative_velocity_tang
        relative_velocity_norm_abs = np.sum(np.abs(relative_velocity_norm)**2,axis=0)**(1./2)
        relative_velocity_unit_norm = np.where(relative_velocity_norm_abs != 0, relative_velocity_norm / relative_velocity_norm_abs, 0) 
        # 液面至構件質心垂直距離
        delta_h = self.OCEAN.eta - center_position[2,:]
        
        # 液面下構件幾何
        under_water_height, section_area, skin_surface  = self.func(delta_h)
        
        # 浸沒體積
        volume_in_water = section_area*element_length

        # 流阻力受力面積
        area_norm = under_water_height*element_length
        area_tang = skin_surface*element_length

        # 計算流阻力係數  依blevins經驗公式
        afa = np.arccos( np.einsum('ij,ij->j', relative_velocity_unit, relative_velocity_unit_tang) )
        cd_norm = 1.2*(np.sin(afa))**2
        cd_tang = 0.083*np.cos(afa)-0.035*(np.cos(afa))**2

        afa = np.where(afa > math.pi/2, math.pi - afa, afa  )

        # if(afa > math.pi/2 ):  #?????????????????????????????
        #     element_tang_unitvector = -element_tang_unitvector

        water_acceleration_abs = np.sum(np.abs(water_acceleration)**2,axis=0)**(1./2)
        judge_water_acceleration = np.where( water_acceleration_abs==0, 0, 1)

        # =============================流阻力 (計算切線及法線方向流阻力分量) ===========================================
        self.flow_resistance_force = ( 0.5*self.OCEAN.water_density*cd_tang*area_tang*relative_velocity_abs**2*relative_velocity_unit_tang + 
                                       0.5*self.OCEAN.water_density*cd_norm*area_norm*relative_velocity_abs**2*relative_velocity_unit_norm )

        # =========================================== 慣性力 ===========================================
        self.inertial_force = self.OCEAN.water_density*self.intertia_coeff*volume_in_water*water_acceleration

        # ===========================================  附加質量 =========================================== 
        self.added_mass_element = (self.intertia_coeff-1)*self.OCEAN.water_density*volume_in_water*judge_water_acceleration

        # ===========================================  浮力 =========================================== 
        self.buoyancy_force[2, :] = self.OCEAN.water_density*volume_in_water*self.OCEAN.gravity

        # ===========================================  重力 =========================================== 
        self.gravity_force[2, :] = -self.element_mass*self.OCEAN.gravity


        # 計算外傳力 (注意張力方向)
        self.node_mass.fill(0)
        self.pass_force.fill(0)

        # 質點總合力
        for i in range(self.num_node):
            
            element_index = self.get_element_index(i)

            for index in element_index:

                self.node_mass[i] += ( self.element_mass + self.added_mass_element[index] )/2

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

        self.node_force = np.copy(self.pass_force)

        # 連結力
        for connection in self.connections:
            if connection["self_node_condition"] == 1:
                self.node_force[:, connection["self_node"]] += connection["connect_obj"].pass_force[:,connection['connect_obj_node']]


        total_mass = np.sum(self.node_mass)
        total_force = np.sum(self.node_force, axis=1)
        
        moment_global_axes = np.sum( np.cross( self.local_position_vectors.T, self.node_force.T ), axis=0)
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
