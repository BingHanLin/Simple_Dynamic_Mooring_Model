import numpy as np

# ==============================================================================
# Runge Kutta 4th 更新質點位置
# ==============================================================================
# http://physics.bu.edu/py502/lectures3/cmotion.pdf
def runge_kutta4(object_list, present_time, dt, ocean_data):

    dt_list = [0*dt, 0.5*dt, 0.5*dt, 1*dt]
    weight_list = [1, 2, 2, 1]

    for rk_step in range(len(dt_list)):

        for obj in object_list:
            obj.time_marching_dt = dt_list[rk_step]

            if rk_step == 0:
                obj.rk4_node_pos_0, obj.rk4_node_vel_0 = obj.get_rk4_position_velocity()
                obj.rk4_node_pos, obj.rk4_node_vel = obj.get_rk4_position_velocity()
            else:
                obj.rk4_node_pos = obj.rk4_node_pos_0 + obj.pk*dt_list[rk_step]   
                obj.rk4_node_vel = obj.rk4_node_vel_0 + obj.vk*dt_list[rk_step]

            obj.correct_node_pos, obj.correct_node_vel = obj.get_correct_position_velocity()


        for obj in object_list:
            obj.correct_node_rk4pos_rk4vel()


        for obj in object_list:
            obj.cal_element_force(obj.rk4_node_pos, 
                                  obj.rk4_node_vel, 
                                  present_time + dt_list[rk_step],
                                  ocean_data)


        for obj in object_list:
            obj.pk, obj.vk = obj.get_rk4vel_rk4acc()
            if rk_step == 0:
                obj.pk_sum, obj.vk_sum = weight_list[rk_step]*obj.pk, weight_list[rk_step]*obj.vk
            else:
                obj.pk_sum, obj.vk_sum = obj.pk_sum + weight_list[rk_step]*obj.pk, obj.vk_sum + weight_list[rk_step]*obj.vk

    # final ==============================================
    for obj in object_list:
        obj.rk4_node_pos = obj.rk4_node_pos_0 + dt*obj.pk_sum / 6
        obj.rk4_node_vel = obj.rk4_node_vel_0 + dt*obj.vk_sum / 6

        obj.correct_node_pos, obj.correct_node_vel = obj.get_correct_position_velocity()

    for obj in object_list:
        # rigid body don't do  this
        obj.correct_node_rk4pos_rk4vel()

        
    # update position and velocity
    for obj in object_list:
       obj.update_position_velocity()
