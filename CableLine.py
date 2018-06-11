import json
import math
import numpy as np

from Structures import STRUCTURES


class CABLELINE(STRUCTURES):
    '''
    Parameters
    ----------
    name : name of the object

    filename : name of parameter file
    
    ocean : wave current 

    start_node : location of start node in 3D

    end_node : location of end node in 3D
    '''
    
    def __init__(self, name, filename, OCEAN, start_node, end_node, CABLETYPE):
        np.seterr(divide='ignore', invalid='ignore') 
        
        with open(filename, 'r') as load_f:
            params_dict = json.load(load_f)

        self.cable_strength = params_dict[CABLETYPE]["ClCable"]
        self.cable_diameter = params_dict[CABLETYPE]["DiameterCable"]
        self.mass_per_length = params_dict[CABLETYPE]["MassPerMCable"]
        self.density = params_dict[CABLETYPE]["MaterialDensity"]
        self.intertia_coeff = params_dict[CABLETYPE]["InertiaCoefficient"]
        self.num_element = params_dict[CABLETYPE]["Elements"]

        self.__cal_const_var()
        self.__init_element(np.asarray(start_node), np.asarray(end_node))

        super().__init__(OCEAN, name, 1)

        print("Cable instance is built.")

    # =======================================
    # 計算繫纜相關常數
    # =======================================
    def __cal_const_var(self):

        self.num_node = self.num_element + 1

        self.element_mass = np.zeros(self.num_element)

    # =======================================
    # 計算初始浮框構件(兩端)質點位置、速度
    # =======================================
    def __init_element(self, start_node, end_node):

        origin_length = np.linalg.norm(end_node - start_node)
        self.origin_element_length =  origin_length / self.num_element

        # global position, velocity, force of nodes
        self.global_node_position = np.zeros((3, self.num_node))
        self.global_node_velocity = np.zeros((3, self.num_node))

        self.node_mass = np.zeros(self.num_node)

        # 初始化浮力、重力
        self.buoyancy_force =  np.zeros((3,self.num_element))
        self.gravity_force =  np.zeros((3,self.num_element))

        # 初始化外傳力
        self.pass_force = np.zeros((3,self.num_node))

        for i in range(self.num_node):
            self.global_node_position[:, i] = (
                end_node - start_node) * i / (self.num_node - 1) + start_node

    # =======================================
    # 計算cd
    # =======================================
    def cal_cd(self, Re_tang, Re_norm):

        if (Re_norm <= 0.1):
            cd_norm = 0
        elif (Re_norm <= 400):
            cd_norm = 0.45+5.93/(Re_norm)**0.33
        elif (Re_norm <= 100000):
            cd_norm = 1.27
        else:
            cd_norm = 0.3


        if (Re_tang <= 0.1):
            cd_tang = 0   
        elif (Re_tang <= 100.55):
            cd_tang = 1.88/(Re_tang)**0.74
        else:
            cd_tang = 0.062


        return cd_tang, cd_norm

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
        
        self.global_node_position = np.copy(global_node_position_temp)
        self.global_node_velocity = np.copy(global_node_velocity_temp)
        
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

        # 構件單位法線相對速度
        relative_velocity_norm  = relative_velocity - relative_velocity_tang
        relative_velocity_norm_abs = np.sum(np.abs(relative_velocity_norm)**2,axis=0)**(1./2)

        Re_tang = relative_velocity_tang_abs*self.cable_diameter/self.OCEAN.water_viscosity
        Re_norm = relative_velocity_norm_abs*self.cable_diameter/self.OCEAN.water_viscosity

        # 計算cd
        cd = np.asarray(list( map(self.cal_cd, Re_tang, Re_norm)))
        cd_tang = cd[:,0]
        cd_norm = cd[:,1]

        area_norm = self.cable_diameter*element_length   
        area_tang = self.cable_diameter*element_length

        epsolon = (element_length - self.origin_element_length) / self.origin_element_length
        epsolon_matrix = np.where( epsolon>0, epsolon, 0)

        water_acceleration_abs = np.sum(np.abs(water_acceleration)**2,axis=0)**(1./2)
        judge_water_acceleration = np.where( water_acceleration_abs==0, 1, 0)

        # =============================流阻力 (計算切線及法線方向流阻力分量) ===========================================
        self.flow_resistance_force = ( 0.5*self.OCEAN.water_density*relative_velocity_tang*(cd_tang*area_tang*relative_velocity_tang_abs) + 
                                       0.5*self.OCEAN.water_density*relative_velocity_norm*(cd_norm*area_norm*relative_velocity_norm_abs) )

        # =========================================== 慣性力 ===========================================
        volume_in_water = self.mass_per_length*element_length/self.density 
        self.inertial_force = self.OCEAN.water_density*self.intertia_coeff*water_acceleration*volume_in_water

        # ===========================================  附加質量 =========================================== 
        self.added_mass_element = (self.intertia_coeff-1)*self.OCEAN.water_density*volume_in_water*judge_water_acceleration

        # ===========================================  浮力 =========================================== 
        self.buoyancy_force[2, :] = self.OCEAN.water_density*volume_in_water*self.OCEAN.gravity

        # ===========================================  重力 =========================================== 
        self.element_mass = self.mass_per_length*element_length
        self.gravity_force[2, :] = - self.element_mass*self.OCEAN.gravity

        # ===========================================  張力 =========================================== 
        self.tension_force = (0.25*math.pi*self.cable_diameter**2)*self.cable_strength*self.OCEAN.gravity*element_tang_unitvector*epsolon_matrix


        # 計算外傳力 (注意張力方向)
        self.node_mass.fill(0)
        self.pass_force.fill(0)

        for i in range(self.num_node):

            if i == 0:
                sign = -1
            else:
                sign = 1

            element_index = self.get_element_index(i)

            for index in element_index:
                sign *= -1
                
                self.node_mass[i] += ( self.element_mass[index] + self.added_mass_element[index] )/2

                self.pass_force[:, i] += (   
                                              self.flow_resistance_force[:, index]/2
                                            + self.inertial_force[:, index]/2
                                            + self.buoyancy_force[:, index]/2                                  
                                            + self.gravity_force[:, index]/2
                                            + self.tension_force[:, index]*sign   
                                         )

    # =======================================
    # 計算回傳速度、加速度
    # =======================================
    def cal_vel_acc(self):

        # 質點總合力
        self.node_force = np.copy(self.pass_force)

        # 連結力

        for connection in self.connections:
            if connection["self_node_condition"] == 1:
                self.node_force[:, connection["self_node"]] += connection["connect_obj"].pass_force[:,connection['connect_obj_node']]



        # 加速度
        global_node_acc_temp = np.where(self.node_mass == 0, 0,  self.node_force / self.node_mass)

        global_node_velocity_temp = np.copy(self.global_node_velocity)

        return global_node_velocity_temp, global_node_acc_temp


    # =======================================
    # 質點 構件 關係
    # =======================================

    def get_node_index(self, element_number):

        node_index = [element_number, element_number+1]

        return node_index


    def get_element_index(self, node_number):

        if node_number  == 0:
            element_index = [0]

        elif node_number == self.num_node-1:
            element_index = [-1]

        else:
            element_index = [node_number-1, node_number]

        return element_index


if __name__ == "__main__":
     pass
