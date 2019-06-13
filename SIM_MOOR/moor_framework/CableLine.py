import json
import math
import numpy as np
from .non_rigid_body import NonRigidBody

class CableLine(NonRigidBody):
    '''
    Parameters
    ----------
    name : name of the object

    start_node : location of start node in 3D

    end_node : location of end node in 3D

    param_dict_name : name of the param dict
    '''
    # ==============================================================================
    # 引入材質參數
    # ==============================================================================
    def load_property(self, param_json_file):
        np.seterr(divide='ignore', invalid='ignore') 
        with open(param_json_file, 'r') as load_f:
            params_data = json.load(load_f)

        params_dict = params_data[self.param_dict_name]
        
        self.cable_strength  = params_dict["ClCable"]
        self.cable_diameter  = params_dict["DiameterCable"]
        self.density         = params_dict["MaterialDensity"]
        self.intertia_coeff  = params_dict["InertiaCoefficient"]
        self.num_element     = params_dict["Elements"]

        self.mass_per_length = 0.25*self.cable_diameter*self.cable_diameter* self.density
        # self.mass_per_length = params_dict["MassPerMCable"]
    # ==============================================================================
    # 初始化構件與質點
    # ==============================================================================
    def init_node_element(self):

        # 初始化質點數、構件長
        self.num_node = self.num_element + 1
        self.node_mass = np.zeros(self.num_node)

        self.origin_element_length =  ( np.linalg.norm(self.end_node_pos - self.start_node_pos) / 
                                        self.num_element )

        # 初始化外傳力
        self.pass_force = np.zeros((3,self.num_node))
        self.node_force = np.zeros((3,self.num_node))

        # 初始化質點位置、速度
        self.node_pos = np.zeros((3, self.num_node))
        self.node_vel = np.zeros((3, self.num_node))


        for i in range(self.num_node):
            self.node_pos[:, i] = self.start_node_pos + ( (self.end_node_pos - self.start_node_pos) * i / 
                                                          (self.num_node - 1) )
        
        # node_index = np.asarray(list( map(self.get_node_index, range(self.num_element))))

        # element_tang_vecotr = self.node_pos[:,node_index[:,1]] - self.node_pos[:,node_index[:,0]]

        # self.origin_element_length = np.sum(np.abs(element_tang_vecotr)**2,axis=0)**(1./2)

    # ==============================================================================
    # 計算cd
    # ==============================================================================
    def cal_cd(self, Re_tang, Re_norm):

        if (Re_tang <= 0.1):
            cd_tang = 0   
        elif (Re_tang <= 100.55):
            cd_tang = 1.88/(Re_tang)**0.74
        else:
            cd_tang = 0.062


        if (Re_norm <= 0.1):
            cd_norm = 0
        elif (Re_norm <= 400):
            cd_norm = 0.45+5.93/(Re_norm)**0.33
        elif (Re_norm <= 100000):
            cd_norm = 1.27
        else:
            cd_norm = 0.3


        return cd_tang, cd_norm

    # ==============================================================================
    # 計算構件受力
    # ==============================================================================
    def cal_element_force(self, node_pos_temp, node_vel_temp, present_time, ocean_data):

        # 迭代構件
        node_index = np.asarray(list( map(self.get_node_index, range(self.num_element))))

        element_pos = 0.5*( node_pos_temp[:,node_index[:,0]] + node_pos_temp[:,node_index[:,1]] )
        element_vel = 0.5*( node_vel_temp[:,node_index[:,0]] + node_vel_temp[:,node_index[:,1]] )

        water_vel, water_acceleration = ocean_data.cal_wave_field(element_pos, present_time)

        relative_vel = water_vel - element_vel
        
        # 構件切線向量
        element_tang_vecotr = node_pos_temp[:,node_index[:,1]] - node_pos_temp[:,node_index[:,0]]
        element_length = np.sum(np.abs(element_tang_vecotr)**2,axis=0)**(1./2)
        element_tang_unitvector  = np.where(element_length != 0, element_tang_vecotr / element_length, 0) 

        # 構件單位切線相對速度
        relative_vel_tang  = np.sum(element_tang_unitvector*relative_vel,axis=0)*element_tang_unitvector
        relative_vel_tang_abs = np.sum(np.abs(relative_vel_tang)**2,axis=0)**(1./2)

        # 構件單位法線相對速度
        relative_vel_norm  = relative_vel - relative_vel_tang
        relative_vel_norm_abs = np.sum(np.abs(relative_vel_norm)**2,axis=0)**(1./2)

        Re_tang = relative_vel_tang_abs*self.cable_diameter/ocean_data.water_viscosity
        Re_norm = relative_vel_norm_abs*self.cable_diameter/ocean_data.water_viscosity

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
        flow_resistance_force = ( 0.5*ocean_data.water_density*relative_vel_tang*(cd_tang*area_tang*relative_vel_tang_abs) + 
                                       0.5*ocean_data.water_density*relative_vel_norm*(cd_norm*area_norm*relative_vel_norm_abs) )

        # =========================================== 慣性力 ===========================================
        # volume_in_water = self.mass_per_length*element_length/self.density 
        volume_in_water = self.mass_per_length*self.origin_element_length/self.density 
        inertial_force = ocean_data.water_density*self.intertia_coeff*water_acceleration*volume_in_water

        # ===========================================  附加質量 =========================================== 
        added_mass_element = (self.intertia_coeff-1)*ocean_data.water_density*volume_in_water*judge_water_acceleration

        # ===========================================  浮力 =========================================== 
        buoyancy_force = np.zeros((3,self.num_element))
        buoyancy_force[2, :] = ocean_data.water_density*volume_in_water*ocean_data.gravity

        # ===========================================  重力 =========================================== 
        element_mass = self.mass_per_length*element_length
        gravity_force = np.zeros((3,self.num_element))
        gravity_force[2, :] = - element_mass*ocean_data.gravity

        # ===========================================  張力 =========================================== 
        tension_force = (0.25*math.pi*self.cable_diameter**2)*self.cable_strength*element_tang_unitvector*epsolon_matrix

        # ===========================================  阻尼力 =========================================== 
        temp_a = node_pos_temp[:,node_index[:,1]] - node_pos_temp[:,node_index[:,0]]
        temp_b = node_vel_temp[:,node_index[:,1]] - node_vel_temp[:,node_index[:,0]]
        temp_c = np.multiply(temp_a, temp_b)
        temp_d = np.sum(temp_c,axis=0)
        # http://www.matt-hall.ca/files/MoorDyn%20Users%20Guide%202015-12-15.pdf
        # https://github.com/mattEhall/MoorDyn/blob/master/Line.cpp
        ldstr = temp_d/element_length
        c = self.origin_element_length*np.sqrt(self.cable_strength*self.mass_per_length / (0.25*math.pi*self.cable_diameter**2)  )
        damping_force = c*(0.25*math.pi*self.cable_diameter**2)*ldstr*element_tang_unitvector

        # print (damping_force)
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
                
                self.node_mass[i] += ( element_mass[index] + added_mass_element[index] )/2

                self.pass_force[:, i] += (   
                                              flow_resistance_force[:, index]/2
                                            + inertial_force[:, index]/2
                                            + buoyancy_force[:, index]/2                                  
                                            + gravity_force[:, index]/2
                                            + tension_force[:, index]*sign
                                            + damping_force[:, index]*sign
                                         )


    # ==============================================================================
    # 質點 構件 關係
    # ==============================================================================
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

    # # ==============================================================================
    # # 計算回傳速度、加速度(test)
    # # ==============================================================================
    # def get_rk4vel_rk4acc(self):

    #     # 質點總合力
    #     self.node_force = np.copy(self.pass_force)

    #     # 加上連結力
    #     for connection in self.connections:
    #         if connection["self_node_condition"] == 1:
    #             self.node_force[:, connection["self_node"]] += connection["connect_obj"].pass_force[:,connection['connect_obj_node']]
    #             self.node_mass[connection["self_node"]] += connection["connect_obj"].node_mass[connection['connect_obj_node']]

    #     # 加速度
    #     node_acc = np.where(self.node_mass == 0, 0,  self.node_force / self.node_mass)

    #     node_acc[:,0] = 0

    #     return self.rk4_node_vel, node_acc

if __name__ == "__main__":
     pass
