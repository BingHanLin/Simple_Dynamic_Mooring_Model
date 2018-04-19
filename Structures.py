import sys
import math
import numpy as np
import pandas as pd
import weakref


class STRUCTURES():

    _instances = set()

    def __init__(self, OCEAN, name, obj_type):

        self.name = name
        self._instances.add(weakref.ref(self))

        self.OCEAN = OCEAN
        self.connections = []
        self.obj_type = obj_type

    # =======================================
    # 增加實例化物件 參考 http://effbot.org/pyfaq/how-do-i-get-a-list-of-all-instances-of-a-given-class.htm
    # ======================================= 
    @classmethod
    def getinstances(cls):
        dead = set()
        for ref in cls._instances:
            obj = ref()
            if obj is not None:
                yield obj
            else:
                dead.add(ref)
        cls._instances -= dead

    # =======================================
    # 增加節點關聯
    # ======================================= 
    def add_conection(self, object, object_node, object_node_condition, self_node):

        self.connections.append({ 'object': object, 
                                  'object_node': object_node,
                                  'object_node_condition': object_node_condition,
                                  'self_node': self_node})
        

    # =======================================
    # Runge Kutta 4th 更新質點位置
    # =======================================
    @classmethod 
    def runge_kutta_4(cls, present_time, dt):

        # 1
        for obj in cls.getinstances():

            if obj.obj_type == 1:
                obj.new_rk4_position = np.copy(obj.global_node_position)     
                obj.new_rk4_velocity = np.copy(obj.global_node_velocity) 

            elif obj.obj_type == 0:
                obj.new_rk4_position =  np.copy(obj.global_center_position) 
                obj.new_rk4_velocity =  np.hstack((obj.global_center_velocity[0:3], obj.local_center_velocity[3:6]))
                # 計算轉換矩陣
                obj.coordinate_transform(dt)

            obj.pk, obj.vk = obj.cal_node_force( present_time, 
                                                 obj.new_rk4_position, 
                                                 obj.new_rk4_velocity)

            obj.pk_sum, obj.vk_sum = obj.pk, obj.vk

        # 2
        for obj in cls.getinstances():

            obj.pk, obj.vk = obj.cal_node_force( present_time + 0.5*dt, 
                                                 obj.new_rk4_position + 0.5*obj.pk*dt , 
                                                 obj.new_rk4_velocity + 0.5*obj.vk*dt)

            obj.pk_sum, obj.vk_sum = obj.pk_sum + 2*obj.pk, obj.vk_sum + 2*obj.vk

        # 3
        for obj in cls.getinstances():

            obj.pk, obj.vk = obj.cal_node_force( present_time + 0.5*dt, 
                                                 obj.new_rk4_position + 0.5*obj.pk*dt , 
                                                 obj.new_rk4_velocity + 0.5*obj.vk*dt)

            obj.pk_sum, obj.vk_sum = obj.pk_sum + 2*obj.pk, obj.vk_sum + 2*obj.vk

        # 4
        for obj in cls.getinstances():

            obj.pk, obj.vk = obj.cal_node_force( present_time + dt, 
                                                 obj.new_rk4_position + obj.pk*dt , 
                                                 obj.new_rk4_velocity + obj.vk*dt)

            obj.pk_sum, obj.vk_sum = obj.pk_sum + obj.pk, obj.vk_sum + obj.vk


        for obj in cls.getinstances():
            obj.new_rk4_position  = obj.new_rk4_position + dt*obj.pk_sum / 6
            obj.new_rk4_velocity =  obj.new_rk4_velocity + dt*obj.vk_sum / 6


    # =======================================
    # 繪出構件
    # ======================================= 
    def plot_element(self, ax):

        if self.num_node == 1:

            ax.scatter(self.global_node_position [0,:], self.global_node_position [1,:], self.global_node_position [2,:], c = 'g')

        else:    
            ax.scatter(self.global_node_position [0,:], self.global_node_position [1,:], self.global_node_position [2,:],c = 'b')

            # for i in range(self.num_node): 
            #     ax.text(self.global_node_position [0, i], self.global_node_position [1, i], self.global_node_position [2, i], str(i),
            #            fontsize=16)

            for i in range(self.num_element): 
                node_index = self.get_node_index(i)

                ax.plot([self.global_node_position [0, node_index[0]], self.global_node_position [0, node_index[1]]],
                        [self.global_node_position [1, node_index[0]], self.global_node_position [1, node_index[1]]],
                        [self.global_node_position [2, node_index[0]], self.global_node_position [2, node_index[1]]],
                        'r')

        axisEqual3D(ax)
        
    # =======================================
    # 儲存結果
    # ======================================= 
    def save_node_data_csv(self, present_time, DirName, save_header):
        
        FileName = './'+ DirName + '/' + 'node_data'+ str("%.5f" % (present_time))+'.csv'


        datafdict = {}

        title_list = ['name','pos_x','pos_y','pos_z'
                            ,'vel_x','vel_y','vel_z'
                            ,'force_x','force_y','forc_z']

        datafdict[title_list[0]] = [self.name]*self.num_node
        datafdict[title_list[1]] =  self.global_node_position[0, :]
        datafdict[title_list[2]] =  self.global_node_position[1, :]
        datafdict[title_list[3]] =  self.global_node_position[2, :]
        datafdict[title_list[4]] =  self.global_node_velocity[0, :]
        datafdict[title_list[5]] =  self.global_node_velocity[1, :]
        datafdict[title_list[6]] =  self.global_node_velocity[2, :]
        datafdict[title_list[7]] =  self.node_force[0,:]
        datafdict[title_list[8]] =  self.node_force[1,:]
        datafdict[title_list[9]] =  self.node_force[2,:]

        dataframe = pd.DataFrame(datafdict)

        if save_header == 0:
            dataframe.to_csv(FileName, sep=',', mode='a', columns = title_list)
        else:
            dataframe.to_csv(FileName, sep=',', mode='a', header=False, columns = title_list)



    def save_element_data_csv(self, present_time, DirName, save_header):
        
        FileName = './'+ DirName + '/' + 'element_data'+ str("%.5f" % (present_time))+'.csv'

        node_index_list = []
        for i in range(self.num_element):
            node_index = self.get_node_index(i)
            node_index_list.append(node_index)   


        datafdict = {}

        title_list = ['name','node_index'
                     ,'tension_force_x','tension_force_y','tension_force_z']

        datafdict[title_list[0]] = [self.name]*self.num_element
        datafdict[title_list[1]] =  node_index_list

        try:
            datafdict[title_list[2]] = self.tension_force[0,:]
            datafdict[title_list[3]] = self.tension_force[1,:]
            datafdict[title_list[4]] = self.tension_force[2,:]
        except:
            datafdict[title_list[2]] = ['nan']*self.num_element
            datafdict[title_list[3]] = ['nan']*self.num_element
            datafdict[title_list[4]] = ['nan']*self.num_element

        dataframe = pd.DataFrame(datafdict)


        if save_header == 0:
            dataframe.to_csv(FileName, sep=',', mode='a')
        else:
            dataframe.to_csv(FileName, sep=',', mode='a', header=False)


# https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio/9349255
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def connect_objects(objects, object_nodes, object_node_conditions):

    if 1 not in object_node_conditions:
        raise ValueError("At least one node condition should be 1")

    print ('============================================')

    for object1, object1_node in zip(objects, object_nodes):
        
        for object2, object2_node, object_node_condition in zip(objects, object_nodes, object_node_conditions):
            
            if object1 != object2:
                object1.add_conection(object2, object2_node, object_node_condition, object1_node)
                print ('Connect %s(index: %d) and %s(index: %d)' % (object1.name, object1_node, object2.name, object2_node))

    print ('============================================')

    
