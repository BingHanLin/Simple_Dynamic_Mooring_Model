# ==============================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import shutil
# ==============================
from Ocean import OCEAN
from Collar import COLLAR
from CableLine import CABLELINE
from BUOY import BUOY
from FRAME import FRAME
from Anchor import ANCHOR
from NET import NET 
from Structures import connect_objects, auto_connect_objects, STRUCTURES

# ============================================================================
# 參數
# ============================================================================
dt = 0.001
display_time = 0.1
save_time = 0.1
total_time = 20

dir_name = 'OUTPUTFRAME_all_net'

# ============================================================================
# 創建物件
# ============================================================================

# 創建波流場
ocean = OCEAN("Params.json")

# 創建整體結構 (注意初始點位)
RECFRAME = FRAME("RECFRAME", "Params.json", ocean, [60.2, 0, -4])

MAINCABLE1 = CABLELINE("MainRope1", "Params.json", ocean, [0, 0, -ocean.water_depth],  RECFRAME.corner_position[:,0], "CABLE")
MAINCABLE2 = CABLELINE("MainRope2", "Params.json", ocean, [0, 20, -ocean.water_depth],  RECFRAME.corner_position[:,1], "CABLE")
MAINCABLE3 = CABLELINE("MainRope3", "Params.json", ocean, [120.4, 20, -ocean.water_depth],  RECFRAME.corner_position[:,2], "CABLE")
MAINCABLE4 = CABLELINE("MainRope4", "Params.json", ocean, [120.4, 0, -ocean.water_depth],  RECFRAME.corner_position[:,3], "CABLE")

BUOY1 = BUOY("BUOY1", "Params.json", ocean, RECFRAME.corner_position[:,0]+[0,0,4])
BUOY2 = BUOY("BUOY2", "Params.json", ocean, RECFRAME.corner_position[:,1]+[0,0,4])
BUOY3 = BUOY("BUOY3", "Params.json", ocean, RECFRAME.corner_position[:,2]+[0,0,4])
BUOY4 = BUOY("BUOY4", "Params.json", ocean, RECFRAME.corner_position[:,3]+[0,0,4])

SUBCABLE1 = CABLELINE("SubRope1", "Params.json", ocean, RECFRAME.corner_position[:,0],  BUOY1.global_node_position[:,0], "SUBCABLE")
SUBCABLE2 = CABLELINE("SubRope2", "Params.json", ocean, RECFRAME.corner_position[:,1],  BUOY2.global_node_position[:,0], "SUBCABLE")
SUBCABLE3 = CABLELINE("SubRope3", "Params.json", ocean, RECFRAME.corner_position[:,2],  BUOY3.global_node_position[:,0], "SUBCABLE")
SUBCABLE4 = CABLELINE("SubRope4", "Params.json", ocean, RECFRAME.corner_position[:,3],  BUOY4.global_node_position[:,0], "SUBCABLE")

# BOTTOMSUBCABLE1 = CABLELINE("BOTTOMSubRope1", "Params.json", ocean, RECFRAME.corner_position[:,0],  RECFRAME.corner_position[:,0]-[0,0,4], "SUBCABLE")
# BOTTOMSUBCABLE2 = CABLELINE("BOTTOMSubRope2", "Params.json", ocean, RECFRAME.corner_position[:,1],  RECFRAME.corner_position[:,1]-[0,0,4], "SUBCABLE")
# BOTTOMSUBCABLE3 = CABLELINE("BOTTOMSubRope3", "Params.json", ocean, RECFRAME.corner_position[:,2],  RECFRAME.corner_position[:,2]-[0,0,4], "SUBCABLE")
# BOTTOMSUBCABLE4 = CABLELINE("BOTTOMSubRope4", "Params.json", ocean, RECFRAME.corner_position[:,3],  RECFRAME.corner_position[:,3]-[0,0,4], "SUBCABLE")

ANCHOR1 = ANCHOR("ANCHOR1", ocean, MAINCABLE1.global_node_position[:,0])
ANCHOR2 = ANCHOR("ANCHOR2", ocean, MAINCABLE2.global_node_position[:,0])
ANCHOR3 = ANCHOR("ANCHOR3", ocean, MAINCABLE3.global_node_position[:,0])
ANCHOR4 = ANCHOR("ANCHOR4", ocean, MAINCABLE4.global_node_position[:,0])

NET1 = NET("NET1", "Params.json", ocean, RECFRAME.global_node_position[:,6:10], np.flip(RECFRAME.global_node_position[:,2:7],1) )

NET2 = NET("NET2", "Params.json", ocean, RECFRAME.global_node_position[:,9:13], np.flip( np.hstack((RECFRAME.global_node_position[:,19:23],
                                                                                                   np.reshape(RECFRAME.global_node_position[:,9],(1,3)).T)) ,1) )


# NET3 = NET("NET3", "Params.json", ocean, RECFRAME.global_node_position[:,2:7], BOTTOMSUBCABLE1.global_node_position[:,0:] )
# NET4 = NET("NET4", "Params.json", ocean, np.flip(RECFRAME.global_node_position[:,12:17],1), BOTTOMSUBCABLE4.global_node_position[:,0:] )

# 建立物件連結 及 錨碇點 (使用節點編號)
connect_objects([RECFRAME, MAINCABLE1, SUBCABLE1], [ RECFRAME.corner_index[0], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE2, SUBCABLE2], [ RECFRAME.corner_index[1], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE3, SUBCABLE3], [ RECFRAME.corner_index[2], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE4, SUBCABLE4], [ RECFRAME.corner_index[3], -1, 0], [1, 0, 0])

# connect_objects([RECFRAME, MAINCABLE1, SUBCABLE1, BOTTOMSUBCABLE1], [ RECFRAME.corner_index[0], -1, 0, 0], [1, 0, 0, 0])
# connect_objects([RECFRAME, MAINCABLE2, SUBCABLE2, BOTTOMSUBCABLE2], [ RECFRAME.corner_index[1], -1, 0, 0], [1, 0, 0, 0])
# connect_objects([RECFRAME, MAINCABLE3, SUBCABLE3, BOTTOMSUBCABLE3], [ RECFRAME.corner_index[2], -1, 0, 0], [1, 0, 0, 0])
# connect_objects([RECFRAME, MAINCABLE4, SUBCABLE4, BOTTOMSUBCABLE4], [ RECFRAME.corner_index[3], -1, 0, 0], [1, 0, 0, 0])

connect_objects([BUOY1, SUBCABLE1], [0, -1], [1, 0])
connect_objects([BUOY2, SUBCABLE2], [0, -1], [1, 0])
connect_objects([BUOY3, SUBCABLE3], [0, -1], [1, 0])
connect_objects([BUOY4, SUBCABLE4], [0, -1], [1, 0])

connect_objects([ANCHOR1, MAINCABLE1], [0, 0], [1, 0])
connect_objects([ANCHOR2, MAINCABLE2], [0, 0], [1, 0])
connect_objects([ANCHOR3, MAINCABLE3], [0, 0], [1, 0])
connect_objects([ANCHOR4, MAINCABLE4], [0, 0], [1, 0])


auto_connect_objects([NET1, RECFRAME], [0,1])
auto_connect_objects([NET2, RECFRAME], [0,1])

# auto_connect_objects([NET3, RECFRAME], [0,1])
# auto_connect_objects([NET3, BOTTOMSUBCABLE1], [0,1])
# auto_connect_objects([NET3, BOTTOMSUBCABLE2], [0,1])

# auto_connect_objects([NET4, RECFRAME], [0,1])
# auto_connect_objects([NET4, BOTTOMSUBCABLE3], [0,1])
# auto_connect_objects([NET4, BOTTOMSUBCABLE4], [0,1])

# for i in range(NET1.num_element):
#     print (i, NET1.get_node_index(i))


# ============================================================================
# Create folder for output data
# ============================================================================
input('please check the above configuration, press any key to continue: ...')

if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
else:
    shutil.rmtree(dir_name)
    os.makedirs(dir_name)

# ======================================

with open(dir_name+'/_info', 'w+') as out_file:
    for obj in STRUCTURES.getinstances():
        out_file.write(obj.name +'\n')


# ============================================================================
# 執行模擬
# ============================================================================

for time_step in range( int(total_time/dt) ):

    present_time = dt*(time_step)

    # 執行 runge_kutta 4th
    STRUCTURES.runge_kutta_4(present_time, dt)
    
    # 更新質點位置及速度
    for obj in STRUCTURES.getinstances():
        obj.update_position_velocity(dt)

    # 存下資料
    if (time_step % (save_time/dt) == 0):
        save_header = 0
        for obj in STRUCTURES.getinstances():
            obj.save_node_data_csv(present_time, dir_name, save_header)
            obj.save_element_data_csv(present_time, dir_name, save_header)
            save_header = 1

    # 顯示計算進度
    if (time_step % (display_time/dt) == 0):
        print ('Time Step: %d, %8.4f sec' % (time_step, present_time))


# ============================================================================
# 繪出結果
# ============================================================================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for obj in STRUCTURES.getinstances():
    obj.plot_element(ax, show_node = True)

plt.show()