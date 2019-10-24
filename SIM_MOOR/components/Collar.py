import json
import math
import numpy as np
from .rigid_body import RigidBody

class Collar(RigidBody):
    '''
    Parameters
    ----------
    name : name of the object

    filename : name of parameter file
    
    ocean : wave current 

    self.start_node_pos : location of start node in 3D (on edge of circle)
    '''
    # ==============================================================================
    # 引入材質參數
    # ==============================================================================
    def load_property(self, param_json_file):
        with open(param_json_file, 'r') as load_f:
            params_data = json.load(load_f)

        params_dict = params_data

        self.collar_radius     = params_dict["COLLAR"]["CollarRadius"]
        self.element_diameter  = params_dict["COLLAR"]["TubeDiameter"]
        self.tube_thickness = params_dict["COLLAR"]["TubeThickness"]
        
        self.density        = params_dict["COLLAR"]["MaterialDensity"]
        self.intertia_coeff = params_dict["COLLAR"]["InertiaCoefficient"]

        self.num_element    = params_dict["COLLAR"]["Elements"]
        self.other_mass     = params_dict["COLLAR"]["OtherMass"]

    # ==============================================================================
    # 初始化構件與質點
    # ==============================================================================
    def init_node_element(self):
        self.bryant_angle = np.zeros(3)

        self.num_node = self.num_element

        # circumference of the collar
        self.collar_circum = 2*math.pi*self.collar_radius

        # total mass of the collar
        total_mass = ( 0.25 * math.pi*
                       ( self.element_diameter**2 - 
                         (self.element_diameter - self.tube_thickness * 2)**2
                       ) *self.collar_circum* self.density + self.other_mass
                     )

        # moment of inertia
        self.moment_inertia = np.zeros(3)
        self.moment_inertia[0] = total_mass*self.collar_radius**2/2
        self.moment_inertia[1] = total_mass*self.collar_radius**2/2
        self.moment_inertia[2] = total_mass*self.collar_radius**2
	    
        # mass of each element
        self.element_mass = total_mass / self.num_element

        self.func = np.vectorize(self.cal_element_under_water)

        # *************************************************************
        # position and velocity of nodes
        self.node_pos = np.zeros((3,self.num_node))
        self.node_vel = np.zeros((3,self.num_node))
        self.node_mass = np.zeros(self.num_node)

        # angle of each element
        theta_element = 2*np.pi/ self.num_element

        for i in range(self.num_node):
            self.node_pos[0, i] = (  self.start_node_pos[0]
                                   + self.collar_radius*np.cos(theta_element*i-np.pi)
                                   + self.collar_radius 
                                  )
            self.node_pos[1, i] =  self.start_node_pos[1] - self.collar_radius*np.sin(theta_element*i-np.pi)
            self.node_pos[2, i] =  self.start_node_pos[2]

        # center position of collar
        self.obj_pos = np.zeros(3)
        self.obj_pos[0] = self.start_node_pos[0] + self.collar_radius
        self.obj_pos[1] = self.start_node_pos[1]
        self.obj_pos[2] = self.start_node_pos[2]  

        # center velocity of collar 
        # local rotational velocity of collar
        self.obj_vel = np.zeros(3)
        self.local_obj_omega = np.zeros(3) 

        # local position vectors of nodes
        self.obj_node_vector = np.zeros((3,self.num_node))

        for i in range(self.num_node):
            self.obj_node_vector[:, i] = self.node_pos[:, i] - self.obj_pos[:]

        # 初始化外傳力
        self.pass_force = np.zeros((3,self.num_node))
        self.node_force = np.zeros((3,self.num_node))
        
        # 初始化浮力、重力
        self.buoyancy_force =  np.zeros((3,self.num_element))
        self.gravity_force =  np.zeros((3,self.num_element))

    # ============================================================================
    # 計算浸水面積
    # ============================================================================
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

    # ============================================================================
    # 計算構件受力
    # ============================================================================
    def cal_element_force(self, obj_position_temp, combine_obj_velocity_temp, present_time, ocean_data):
        
        node_pos_temp, node_vel_temp =  self.trans_pos_vel( obj_position_temp, combine_obj_velocity_temp)
        
        # 迭代構件
        node_index = np.asarray(list( map(self.get_node_index, range(self.num_element))))

        element_position = 0.5*( node_pos_temp[:,node_index[:,0]] + node_pos_temp[:,node_index[:,1]] )
        element_velocity = 0.5*( node_vel_temp[:,node_index[:,0]] + node_vel_temp[:,node_index[:,1]] )

        water_velocity, water_acceleration = ocean_data.cal_wave_field(element_position, present_time)

        relative_velocity = water_velocity - element_velocity

        # 構件切線向量
        element_tang_vecotr = self.node_pos[:,node_index[:,1]] - self.node_pos[:,node_index[:,0]]

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
        delta_h = ocean_data.eta - element_position[2,:]
        
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

        water_acceleration_abs = np.sum(np.abs(water_acceleration)**2,axis=0)**(1./2)
        judge_water_acceleration = np.where( water_acceleration_abs==0, 1, 0)

        # =============================流阻力 (計算切線及法線方向流阻力分量) ===========================================
        self.flow_resistance_force = ( 0.5*ocean_data.water_density*cd_tang*area_tang*relative_velocity_abs**2*relative_velocity_unit_tang + 
                                       0.5*ocean_data.water_density*cd_norm*area_norm*relative_velocity_abs**2*relative_velocity_unit_norm )

        # =========================================== 慣性力 ===========================================
        self.inertial_force = ocean_data.water_density*self.intertia_coeff*volume_in_water*water_acceleration

        # ===========================================  附加質量 =========================================== 
        self.added_mass_element = (self.intertia_coeff-1)*ocean_data.water_density*volume_in_water*judge_water_acceleration

        # ===========================================  浮力 =========================================== 
        self.buoyancy_force[2, :] = ocean_data.water_density*volume_in_water*ocean_data.gravity

        # ===========================================  重力 =========================================== 
        self.gravity_force[2, :] = -self.element_mass*ocean_data.gravity


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

    # ============================================================================
    # 質點 構件 關係
    # ============================================================================
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


if __name__ == "__main__":
    pass
