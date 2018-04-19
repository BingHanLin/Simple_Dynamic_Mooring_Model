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
from Structures import connect_objects, STRUCTURES

# ============================================================================
# 參數
# ============================================================================
dt = 0.001
display_time = 0.01
save_time = 0.05
total_time = 15

dir_name = 'OUTPUTFRAME'

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

# 建立物件連結 及 錨碇點 (使用節點編號)
connect_objects([RECFRAME, MAINCABLE1, SUBCABLE1], [ RECFRAME.corner_index[0], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE2, SUBCABLE2], [ RECFRAME.corner_index[1], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE3, SUBCABLE3], [ RECFRAME.corner_index[2], -1, 0], [1, 0, 0])
connect_objects([RECFRAME, MAINCABLE4, SUBCABLE4], [ RECFRAME.corner_index[3], -1, 0], [1, 0, 0])

connect_objects([BUOY1, SUBCABLE1], [0, -1], [1, 0])
connect_objects([BUOY2, SUBCABLE2], [0, -1], [1, 0])
connect_objects([BUOY3, SUBCABLE3], [0, -1], [1, 0])
connect_objects([BUOY4, SUBCABLE4], [0, -1], [1, 0])

MAINCABLE1.add_anchor(0)
MAINCABLE2.add_anchor(0)
MAINCABLE3.add_anchor(0)
MAINCABLE4.add_anchor(0)

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

    # 顯示計算進度
    if (time_step % (display_time/dt) == 0):
        print ('Time Step: %d, %8.4f sec' % (time_step, present_time))


# ============================================================================
# 繪出結果
# ============================================================================
# for i in range(RECFRAME.num_node):

#     print (i , RECFRAME.get_element_index(i))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ocean.plot_ocean([-10,130], [-30,50], 50, 50, 0, ax)

for obj in STRUCTURES.getinstances():
    obj.plot_element(ax)

plt.show()