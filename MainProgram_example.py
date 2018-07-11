# ==============================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import shutil
# ==============================
from Ocean import OCEAN
from CableLine import CABLELINE
from BUOY import BUOY
from FRAME import FRAME
from Anchor import ANCHOR
from WEIGHTING import WEIGHTING
from NET import NET
from Structures import auto_connect_objects, connect_objects, STRUCTURES

# ============================================================================
# 參數
# ============================================================================
dt = 0.0005
display_time = 0.01
save_time = 0.05
total_time = 40

dir_name = 'hanging0605'

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


NET1 = NET("NET1", "Params.json", ocean, RECFRAME.global_node_position[:,6:10], np.flip(RECFRAME.global_node_position[:,2:7],1) )

NET2 = NET("NET2", "Params.json", ocean, RECFRAME.global_node_position[:,9:13], np.flip( np.hstack((RECFRAME.global_node_position[:,19:23],
                                                                                                   np.reshape(RECFRAME.global_node_position[:,9],(1,3)).T)) ,1) )


HANGINGCABLE1 = CABLELINE("HANGINGCABLE1", "Params.json", ocean, NET1.global_node_position[:,13],NET1.global_node_position[:,13]+[0,0,-8], "SUBCABLE")
HANGINGCABLE2 = CABLELINE("HANGINGCABLE2", "Params.json", ocean, NET1.global_node_position[:,9],NET1.global_node_position[:,9]+[0,0,-8], "SUBCABLE")
HANGINGCABLE3 = CABLELINE("HANGINGCABLE3", "Params.json", ocean, NET1.global_node_position[:,5],NET1.global_node_position[:,5]+[0,0,-8], "SUBCABLE")
HANGINGCABLE4 = CABLELINE("HANGINGCABLE4", "Params.json", ocean, NET1.global_node_position[:,6],NET1.global_node_position[:,6]+[0,0,-8], "SUBCABLE")
HANGINGCABLE5 = CABLELINE("HANGINGCABLE5", "Params.json", ocean, NET1.global_node_position[:,10],NET1.global_node_position[:,10]+[0,0,-8], "SUBCABLE")
HANGINGCABLE6 = CABLELINE("HANGINGCABLE6", "Params.json", ocean, NET1.global_node_position[:,14],NET1.global_node_position[:,14]+[0,0,-8], "SUBCABLE")

HANGINGCABLE7 = CABLELINE("HANGINGCABLE7", "Params.json", ocean, NET2.global_node_position[:,13], NET2.global_node_position[:,13]+[0,0,-8],"SUBCABLE")
HANGINGCABLE8 = CABLELINE("HANGINGCABLE8", "Params.json", ocean, NET2.global_node_position[:,9],NET2.global_node_position[:,9]+[0,0,-8], "SUBCABLE")
HANGINGCABLE9 = CABLELINE("HANGINGCABLE9", "Params.json", ocean, NET2.global_node_position[:,5],NET2.global_node_position[:,5]+[0,0,-8], "SUBCABLE")
HANGINGCABLE10 = CABLELINE("HANGINGCABLE10", "Params.json", ocean, NET2.global_node_position[:,6],NET2.global_node_position[:,6]+[0,0,-8], "SUBCABLE")
HANGINGCABLE11 = CABLELINE("HANGINGCABLE11", "Params.json", ocean, NET2.global_node_position[:,10],NET2.global_node_position[:,10]+[0,0,-8], "SUBCABLE")
HANGINGCABLE12 = CABLELINE("HANGINGCABLE12", "Params.json", ocean, NET2.global_node_position[:,14],NET2.global_node_position[:,14]+[0,0,-8], "SUBCABLE")

WEIGHTING1 = WEIGHTING("WEIGHTING1", "Params.json", ocean, HANGINGCABLE1.global_node_position[:, -1])
WEIGHTING2 = WEIGHTING("WEIGHTING2", "Params.json", ocean, HANGINGCABLE2.global_node_position[:, -1])
WEIGHTING3 = WEIGHTING("WEIGHTING3", "Params.json", ocean, HANGINGCABLE3.global_node_position[:, -1])
WEIGHTING4 = WEIGHTING("WEIGHTING4", "Params.json", ocean, HANGINGCABLE4.global_node_position[:, -1])
WEIGHTING5 = WEIGHTING("WEIGHTING5", "Params.json", ocean, HANGINGCABLE5.global_node_position[:, -1])
WEIGHTING6 = WEIGHTING("WEIGHTING6", "Params.json", ocean, HANGINGCABLE6.global_node_position[:, -1])

WEIGHTING7 = WEIGHTING("WEIGHTING7", "Params.json", ocean, HANGINGCABLE7.global_node_position[:, -1])
WEIGHTING8 = WEIGHTING("WEIGHTING8", "Params.json", ocean, HANGINGCABLE8.global_node_position[:, -1])
WEIGHTING9 = WEIGHTING("WEIGHTING9", "Params.json", ocean, HANGINGCABLE9.global_node_position[:, -1])
WEIGHTING10 = WEIGHTING("WEIGHTING10", "Params.json", ocean, HANGINGCABLE10.global_node_position[:, -1])
WEIGHTING11 = WEIGHTING("WEIGHTING11", "Params.json", ocean, HANGINGCABLE11.global_node_position[:, -1])
WEIGHTING12 = WEIGHTING("WEIGHTING12", "Params.json", ocean, HANGINGCABLE12.global_node_position[:, -1])


ANCHOR1 = ANCHOR("ANCHOR1", ocean, MAINCABLE1.global_node_position[:,0])
ANCHOR2 = ANCHOR("ANCHOR2", ocean, MAINCABLE2.global_node_position[:,0])
ANCHOR3 = ANCHOR("ANCHOR3", ocean, MAINCABLE3.global_node_position[:,0])
ANCHOR4 = ANCHOR("ANCHOR4", ocean, MAINCABLE4.global_node_position[:,0])



# 建立物件連結 及 錨碇點 (使用節點編號)
connect_objects([RECFRAME, MAINCABLE1, SUBCABLE1], [ RECFRAME.corner_index[0], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE2, SUBCABLE2], [ RECFRAME.corner_index[1], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE3, SUBCABLE3], [ RECFRAME.corner_index[2], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE4, SUBCABLE4], [ RECFRAME.corner_index[3], -1, 0], [1, 0, 0])

connect_objects([HANGINGCABLE1, NET1], [ 0, 13], [0, 1])
connect_objects([HANGINGCABLE2, NET1], [ 0, 9], [0, 1])
connect_objects([HANGINGCABLE3, NET1], [ 0, 5], [0, 1])
connect_objects([HANGINGCABLE4, NET1], [ 0, 6], [0, 1])
connect_objects([HANGINGCABLE5, NET1], [ 0, 10], [0, 1])
connect_objects([HANGINGCABLE6, NET1], [ 0, 14], [0, 1])

connect_objects([HANGINGCABLE7, NET2], [ 0, 13], [0, 1])
connect_objects([HANGINGCABLE8, NET2], [ 0, 9], [0, 1])
connect_objects([HANGINGCABLE9, NET2], [ 0, 5], [0, 1])
connect_objects([HANGINGCABLE10, NET2], [ 0, 6], [0, 1])
connect_objects([HANGINGCABLE11, NET2], [ 0, 10], [0, 1])
connect_objects([HANGINGCABLE12, NET2], [ 0, 14], [0, 1])

connect_objects([HANGINGCABLE1, WEIGHTING1], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE2, WEIGHTING2], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE3, WEIGHTING3], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE4, WEIGHTING4], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE5, WEIGHTING5], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE6, WEIGHTING6], [ -1, 0], [0, 1])

connect_objects([HANGINGCABLE7, WEIGHTING7], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE8, WEIGHTING8], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE9, WEIGHTING9], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE10, WEIGHTING10], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE11, WEIGHTING11], [ -1, 0], [0, 1])
connect_objects([HANGINGCABLE12, WEIGHTING12], [ -1, 0], [0, 1])

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


# ============================================================================
# Create folder for output data
# ============================================================================

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

        ocean.save_data_csv(present_time, dir_name)

   # 顯示計算進度
   if (time_step % (display_time/dt) == 0):
       print ('Time Step: %d, %8.4f sec' % (time_step, present_time))


# ============================================================================
# 繪出結果
# ============================================================================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for obj in STRUCTURES.getinstances():
    obj.plot_element(ax)

plt.show()