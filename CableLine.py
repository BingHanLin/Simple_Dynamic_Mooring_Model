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

        
        with open(filename, 'r') as load_f:
            params_dict = json.load(load_f)

        self.cable_strength = params_dict[CABLETYPE]["ClCable"]
        self.cable_diameter = params_dict[CABLETYPE]["DiameterCable"]
        self.mass_per_length = params_dict[CABLETYPE]["MassPerMCable"]
        self.density = params_dict[CABLETYPE]["MaterialDensity"]
        self.intertia_coeff = params_dict[CABLETYPE]["InertiaCoefficient"]
        self.num_element = params_dict[CABLETYPE]["Elements"]

        self.anchor_node =[]

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
        self.pass_force = np.zeros((3,self.num_node))

        for i in range(self.num_node):
            self.global_node_position[:, i] = (
                end_node - start_node) * i / (self.num_node - 1) + start_node

    # =======================================
    # 增錨碇點
    # ======================================= 
    def add_anchor(self, node):

        self.anchor_node.append(node)

    # =======================================
    # Runge Kutta 4th 更新質點位置
    # =======================================

    def update_position_velocity(self, dt):
        
        for anchor_node_index in self.anchor_node:
            self.new_rk4_position[:,anchor_node_index] = self.global_node_position[:, anchor_node_index]
            self.new_rk4_velocity[:,anchor_node_index] = np.zeros(3)


        for connection in self.connections:
            if connection["object_node_condition"] == 1:
                self.new_rk4_position[:,connection["self_node"]] = connection["object"].global_node_position[:,connection['object_node']]
                self.new_rk4_velocity[:,connection["self_node"]] = connection["object"].global_node_velocity[:,connection['object_node']]


        self.global_node_position = np.copy(self.new_rk4_position)
        self.global_node_velocity = np.copy(self.new_rk4_velocity)


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
    # 計算構件受力
    # =======================================
    def cal_node_force(self,  present_time, global_node_position_temp, global_node_velocity_temp):
        
        for connection in self.connections:
            if connection["object_node_condition"] == 1:
                global_node_position_temp[:,connection["self_node"]] = connection["object"].global_node_position[:,connection['object_node']]
                global_node_velocity_temp[:,connection["self_node"]] = connection["object"].global_node_velocity[:,connection['object_node']]

        for anchor_node_index in self.anchor_node:
            global_node_position_temp[:,anchor_node_index] = global_node_position_temp[:, anchor_node_index]
            global_node_velocity_temp[:,anchor_node_index] = np.zeros(3)

        # 初始化流阻力、慣性力、浮力、重力、附加質量
        flow_resistance_force =  np.zeros((3,self.num_element))
        inertial_force =  np.zeros((3,self.num_element))
        buoyancy_force =  np.zeros((3,self.num_element))
        gravity_force =  np.zeros((3,self.num_element))
        self.tension_force = np.zeros((3,self.num_element))
        connected_force = np.zeros((3,self.num_node))
        added_mass_element = np.zeros(self.num_element)
        self.pass_force = np.zeros((3,self.num_node))
        
        # 初始化質點集中受力、質點集中質量
        self.node_force = np.zeros((3,self.num_node))
        node_mass = np.zeros(self.num_node)
        global_node_acc_temp = np.zeros((3,self.num_node))


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
            relative_velocity_tang  = np.dot(element_tang_unitvector,relative_velocity)*element_tang_unitvector
            relative_velocity_tang_abs = np.linalg.norm(relative_velocity_tang)

            # 構件單位法線相對速度
            relative_velocity_norm = relative_velocity - relative_velocity_tang
            relative_velocity_norm_abs = np.linalg.norm(relative_velocity_norm)

            Re_tang = relative_velocity_tang_abs*self.cable_diameter/self.OCEAN.water_viscosity
            Re_norm = relative_velocity_norm_abs*self.cable_diameter/self.OCEAN.water_viscosity

            cd_tang, cd_norm = self.cal_cd(Re_tang, Re_norm)

            area_norm = self.cable_diameter*element_length    
            area_tang = self.cable_diameter*element_length   #???????

 
            # 流阻力 (計算切線及法線方向流阻力分量)
            flow_resistance_force_tang =  0.5*self.OCEAN.water_density*cd_tang*area_tang*relative_velocity_tang_abs*relative_velocity_tang

            flow_resistance_force_norm =  0.5*self.OCEAN.water_density*cd_norm*area_norm*relative_velocity_norm_abs*relative_velocity_norm

            flow_resistance_force[:, i] = flow_resistance_force_tang + flow_resistance_force_norm
            
            # 慣性力
            volume_in_water = self.mass_per_length*element_length/self.density 

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
            self.element_mass[i] = self.mass_per_length*element_length
            gravity_force[:, i] = np.asarray([   0,
                                                 0, 
                                                - self.element_mass[i]*self.OCEAN.gravity])


            # 張力
            epsolon = (element_length - self.origin_element_length) / self.origin_element_length

            if epsolon > 0:
                self.tension_force[:, i] = (0.25*math.pi*self.cable_diameter**2)*self.cable_strength*epsolon*self.OCEAN.gravity*element_tang_unitvector
            else:
                self.tension_force[:, i] = np.zeros(3)


        # 連結力
        for connection in self.connections:
            if connection["object_node_condition"] != 1:
                connected_force[:, connection["self_node"]] += connection["object"].pass_force[:,connection['object_node']]


        # 質點總合力 (注意張力方向)
        for i in range(self.num_node):
            if i == 0:
                sign = -1
            else:
                sign = 1

            element_index = self.get_element_index(i)

            for index in element_index:
                sign *= -1
                self.pass_force[:, i] += (   flow_resistance_force[:, index]
                                            + inertial_force[:, index]
                                            + buoyancy_force[:, index]
                                            + self.tension_force[:, index]*sign                                     
                                            + gravity_force[:, index] )
            
                node_mass[i] += ( self.element_mass[index] + added_mass_element[index] )


            self.pass_force[:, i] = self.pass_force[:, i]/len(element_index)

            self.node_force[:,i] = self.pass_force[:,i] + connected_force[:,i]

            node_mass[i] = node_mass[i]/len(element_index)

  
            if (node_mass[i] == 0):
                global_node_acc_temp[:,i] = 0
            else:
                global_node_acc_temp[:,i] = self.node_force[:,i]/node_mass[i]


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
