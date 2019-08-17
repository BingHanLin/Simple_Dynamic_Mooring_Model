from SIM_MOOR import Model
from SIM_MOOR.moor_framework import CableLine, Anchor, Collar
# ==============================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

# ============================================================================
# 參數
# ============================================================================
delta_t = 0.001
display_time_steps = 1000
save_time_steps = 1000
num_time_steps = 50000
dir_name = 'hanging0605'

# ============================================================================
# 創建錨定模型
# ============================================================================
model = Model("Params.json")

model.add(CableLine("MainRope", [60.2, 0, -20], [60.2, 0, -4], "CABLE"))
# model.add(CableLine("MainRope1", [60.2, 0, -10], [60.2, 0, -1], "CABLE"))
model.add(Anchor("Anchor", [60.2, 0, -20]))
model.add(Collar("Collar", [60.2, 0, -4]))
  
model.connect(["MainRope",0, 0], ["Anchor", 1, 0])
# model.connect(["MainRope", 1], ["MainRope1", 0])
model.connect(["MainRope", 0], ["Collar", 1])

model.summary()

# ============================================================================
# 執行模擬
# ============================================================================
model.run(dir_name, delta_t, num_time_steps, save_time_steps, display_time_steps )

