import sys
import math
import numpy as np
import pandas as pd

class FrameWork():

    def __init__(self, name, start_node_pos, end_node_pos= None, param_dict_name = None):
        
        self.name = name
        self.param_dict_name = param_dict_name
       
        self.start_node_pos = np.asarray(start_node_pos)
        self.end_node_pos   = np.asarray(end_node_pos)

        self.connections = []

    # ==============================================================================
    # 增加連接質點
    # ============================================================================== 
    def add_conection(self, connect_obj, connect_obj_node, connect_obj_node_condition, self_node, self_node_condition):

        self.connections.append({ 'connect_obj': connect_obj, 
                                  'connect_obj_node': connect_obj_node,
                                  'connect_obj_node_condition': connect_obj_node_condition,
                                  'self_node': self_node,
                                  'self_node_condition':self_node_condition})




