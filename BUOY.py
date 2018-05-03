import json
import math
import numpy as np

from Structures import STRUCTURES

class BUOY(STRUCTURES):
    def __init__(self, name, filename, OCEAN, start_node):

        with open(filename, 'r') as load_f:
            params_dict = json.load(load_f)

        self.mass      = params_dict["BUOY"]["Mass"]
        self.diameter  = params_dict["BUOY"]["Diameter"]
        self.height = params_dict["BUOY"]["Height"]
        self.intertia_coeff = params_dict["BUOY"]["InertiaCoefficient"]

        self.__cal_const_var()
        self.__init_element( np.asarray(start_node) )

        super().__init__(OCEAN, name, 1)


        print("BUOY is built.")

    # =======================================
    # 計算浮子相關常數
    # =======================================
    def __cal_const_var(self):
        self.element_mass =  [self.mass]
        self.num_element = 1
        self.num_node = 1

    # =======================================
    # 計算初始浮框構件(兩端)質點位置、速度等
    # =======================================
    def __init_element(self, start_node):

        # global position, velocity, force of nodes
        self.global_node_position = np.zeros((3, self.num_node))
        self.global_node_position[:, 0] = start_node
        self.global_node_velocity = np.zeros((3, self.num_node))
        self.pass_force = np.zeros((3,self.num_node))

    # =======================================
    # 更新質點位置
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

        self.weight = self.mass*self.OCEAN.gravity 

        # 初始化流阻力、慣性力、浮力、重力、附加質量
        self.flow_resistance_force =  np.zeros((3,self.num_element))
        self.inertial_force =  np.zeros((3,self.num_element))
        self.buoyancy_force =  np.zeros((3,self.num_element))
        self.gravity_force =  np.zeros((3,self.num_element))
        self.added_mass_element = np.zeros(self.num_element)

        # 初始化外傳力
        self.pass_force = np.zeros((3,self.num_node))

        

        for i in range(self.num_element):
            node_index = self.get_node_index(i)
            # 構件質心處海波流場
            water_velocity, water_acceleration = self.OCEAN.cal_wave_field(
                                                (self.global_node_position[0, node_index[0]] + self.global_node_position[0, node_index[0]])/2,
                                                (self.global_node_position[1, node_index[0]] + self.global_node_position[1, node_index[0]])/2,
                                                (self.global_node_position[2, node_index[0]] + self.global_node_position[2, node_index[0]])/2,
                                                present_time )

            # 構件與海水相對速度
            relative_velocity  = water_velocity - self.global_node_velocity[:,i]
            relative_velocity_abs = np.linalg.norm(relative_velocity)

            if relative_velocity_abs == 0:
                relative_velocity_unit = np.zeros(3)
            else:
                relative_velocity_unit = relative_velocity / relative_velocity_abs
                
            # 構件切線向量
            connection = self.connections[0]
            element_tang_vecotr =  [ self.global_node_position[0,i] - connection["connect_obj"].global_node_position[0,connection['connect_obj_node']-1],
                                     self.global_node_position[1,i] - connection["connect_obj"].global_node_position[1,connection['connect_obj_node']-1],
                                     self.global_node_position[2,i] - connection["connect_obj"].global_node_position[2,connection['connect_obj_node']-1] ]


            element_length = np.linalg.norm(element_tang_vecotr)
            element_tang_unitvector = element_tang_vecotr / element_length

            # 構件單位切線相對速度
            relative_velocity_unit_tang  = np.dot(element_tang_unitvector,relative_velocity_unit)*element_tang_unitvector

            # 構件單位法線相對速度
            relative_velocity_unit_norm = relative_velocity_unit - relative_velocity_unit_tang

            # 計算浮球傾斜角度
            if (element_length != 0):
                theta_b = np.arcsin( abs(element_tang_vecotr[2]) / element_length)
            else:
                raise ValueError("wrong in buoy")


            # 液面至構件質心垂直距離
            dh = self.OCEAN.eta - self.global_node_position[2, i] 

	        # 計算浸水深度 面積 表面積
            if (dh > self.height):
                volume_in_water = ( 0.25*math.pi*self.diameter**2)*self.height

                self.buoyancy_force[:, i] = np.asarray([   0,
                                                      0,
                                                      self.OCEAN.water_density*volume_in_water*self.OCEAN.gravity])

                area_norm = self.height*self.diameter
                area_tang = self.height*self.diameter

            elif (dh < 0):
                volume_in_water = 0

                self.buoyancy_force[:, i] = np.asarray([ 0 ,0, 0])

                area_norm = 0
                area_tang = 0

            else:

                if(np.sin(theta_b) != 0):
                    bh = dh / np.sin(theta_b)
                else:
                    bh = dh
                volume_in_water = ( 0.25*math.pi*self.diameter**2)*self.height
                
                self.buoyancy_force[:, i] = np.asarray([ 0,
                                                         0,
                                                         self.OCEAN.water_density*volume_in_water*self.OCEAN.gravity])

                area_norm = bh*self.diameter
                area_tang = bh*self.diameter*math.pi
                

            # 計算流阻力係數  依blevins經驗公式
            if( abs(np.dot(relative_velocity_unit, element_tang_unitvector)) > 1): 
                afa = 0.5*math.pi
            else:
                afa = np.arccos( np.dot(relative_velocity_unit, element_tang_unitvector) )

            if(afa > math.pi/2 ):
                afa = math.pi - afa
                element_tang_unitvector = -element_tang_unitvector

            cd_norm = 1.2*(np.sin(afa))**2
            cd_tang = 0.083*np.cos(afa)-0.035*(np.cos(afa))**2


            # 流阻力 (計算切線及法線方向流阻力分量)
            flow_resistance_force_tang = 0.5*self.OCEAN.water_density*cd_tang*area_tang*relative_velocity_abs**2*relative_velocity_unit_tang

            flow_resistance_force_norm = 0.5*self.OCEAN.water_density*cd_norm*area_norm*relative_velocity_abs**2*relative_velocity_unit_norm
            
            self.flow_resistance_force[:, i] = 0.5*(flow_resistance_force_tang + flow_resistance_force_norm)
            
            # 慣性力
            self.inertial_force[:, i] = self.OCEAN.water_density*self.intertia_coeff*volume_in_water*water_acceleration

            if ( np.linalg.norm(water_acceleration) != 0):
                self.added_mass_element[i] = (self.intertia_coeff-1)*self.OCEAN.water_density*volume_in_water
            else:
                self.added_mass_element[i] = 0
            
            # 重力
            self.gravity_force[:, i] = np.asarray([ 0,
                                                    0, 
                                                    - self.element_mass[i]*self.OCEAN.gravity])
            
       # 計算外傳力
        for i in range(self.num_node):


            self.pass_force[:,i] = (   self.flow_resistance_force[:, i]
                                     + self.inertial_force[:, i]
                                     + self.buoyancy_force[:, i]                              
                                     + self.gravity_force[:, i]        )
            


    # =======================================
    # 計算回傳速度、加速度
    # =======================================
    def cal_vel_acc(self):

        self.node_force = np.zeros((3,self.num_node))
        node_mass = np.zeros(self.num_node)
        global_node_acc_temp = np.zeros((3,self.num_node))
        connected_force = np.zeros((3,self.num_node))

        # 連結力
        for connection in self.connections:
            if connection["self_node_condition"] == 1:
                connected_force[:, connection["self_node"]] += connection["connect_obj"].pass_force[:,connection['connect_obj_node']]


        # 質點總合力
        for i in range(self.num_node):


            self.node_force[:,i] = (   self.flow_resistance_force[:, i]
                                     + self.inertial_force[:, i]
                                     + self.buoyancy_force[:, i]                              
                                     + self.gravity_force[:, i]        )
            

            self.node_force[:,i]  = self.node_force[:,i] + connected_force[:,i]

            node_mass[i] = (   (self.element_mass[i]) + self.added_mass_element[i]  )


            if (node_mass[i] == 0):
                global_node_acc_temp[:,i] = 0
            else:
                global_node_acc_temp[:,i] = self.node_force[:,i]/node_mass[i]
        
        
        global_node_velocity_temp = np.copy(self.global_node_velocity)

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
