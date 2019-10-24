import itertools
import numpy as np

def connect_objects(*object_list, check_flag = True):
    '''
    Parameters
    ----------
    [ object, node_idx, node_cond],...
     
    object : name of the object

    node_idx : node index of the object
    
    node_cond : define the leader (#1) and the follower (#0) objects.
    '''
    # ==============================================================================
    # create object_dict_list 
    # ==============================================================================
    keys = ['object', 'node_cond', 'node_idx']

    object_dict_list = []

    for each_object in object_list:
        object_dict_list.append(dict( zip( keys, each_object )))

    # ==============================================================================
    # check number of connected objects
    # ==============================================================================
    if len(object_dict_list) < 2:
        raise ValueError("At least two objects should be in arguments.")

    # ==============================================================================
    # check node condition
    # ==============================================================================
    node_condition_list  = [ each_object['node_cond'] for each_object in object_dict_list ] 
   
    if 1 not in node_condition_list and check_flag == True:
        raise ValueError("At least one object should be the leader object (#1).")

    # ==============================================================================
    # connect objects
    # ==============================================================================
    for objectes_pair in itertools.combinations(object_dict_list, 2):

        objectes_pair[0]['object'].add_conection(objectes_pair[1]['object'], 
                                                 objectes_pair[1]['node_idx'], objectes_pair[1]['node_cond'],
                                                 objectes_pair[0]['node_idx'], objectes_pair[0]['node_cond'] )

        objectes_pair[1]['object'].add_conection(objectes_pair[0]['object'], 
                                                 objectes_pair[0]['node_idx'], objectes_pair[0]['node_cond'],
                                                 objectes_pair[1]['node_idx'], objectes_pair[1]['node_cond'] )

        print ('Connect %s(index: %d) and %s(index: %d)' % (objectes_pair[0]['object'].name, objectes_pair[0]['node_idx'], 
                                                            objectes_pair[1]['object'].name, objectes_pair[1]['node_idx']))



def auto_connect_objects(*object_list):
    '''
    Parameters
    ----------
    [ object, node_cond],...
     
    object : name of the object
    
    node_cond : define the leader (#1) and the follower (#0) objects.
    '''
    # ==============================================================================
    # create object_dict_list 
    # ==============================================================================
    keys = ['object', 'node_cond']

    object_dict_list = []

    for each_object in object_list:
        object_dict_list.append(dict( zip( keys, each_object )))

    # ==============================================================================
    # check number of connected objects
    # ==============================================================================
    # if len(object_dict_list) != 2:
    #     raise ValueError("Two objects should be in arguments.")
    if len(object_dict_list) < 2:
        raise ValueError("At least two objects should be in arguments.")
    # ==============================================================================
    # check node condition
    # ==============================================================================
    node_condition_list  = [ each_object['node_cond'] for each_object in object_dict_list ] 
   
    if 1 not in node_condition_list:
        raise ValueError("At least one object should be the leader object (#1).")

    # ==============================================================================
    # connect objects
    # ==============================================================================

    for objectes_pair in itertools.combinations(object_dict_list, 2):

        flag_have_same_node = False

        for node_idx_pair in itertools.product( list(range(objectes_pair[0]['object'].num_node)),
                                                list(range(objectes_pair[1]['object'].num_node)) ):
            

            if (np.allclose( objectes_pair[0]['object'].node_pos[:, node_idx_pair[0]], 
                             objectes_pair[1]['object'].node_pos[:, node_idx_pair[1]])):

                connect_objects( [ objectes_pair[0]['object'], objectes_pair[0]['node_cond'], node_idx_pair[0] ],
                                 [ objectes_pair[1]['object'], objectes_pair[1]['node_cond'], node_idx_pair[1] ],
                                 check_flag = False)

                flag_have_same_node = True

        if flag_have_same_node == False:
            raise ValueError("No same node position for two objects.")


