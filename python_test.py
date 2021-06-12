import torch
import numpy as np
import matplotlib.pyplot as plt


with open('/home/lab/Python_pro/Ly_pro/SSD_tc/checkpoints/model-110-vallosses.npy', 'rb') as val:
    val = np.load(val)
with open('/home/lab/Python_pro/Ly_pro/SSD_tc/checkpoints/model-110-losses.npy', 'rb') as los:
    los = np.load(los)

val = np.nan_to_num(val)
# print(val)
los = np.nan_to_num(los)
# print(val.shape)
figure = plt.figure()
x = np.arange(1, 111)
# plt.xlim(0, 80)
plt.plot(x, val, 'b')
plt.plot(x, los, 'r')
plt.show()
# print((val[73], los[74]))
# dex = 0
# for i in los:
#     dex += 1
#     if np.isnan(i):
#         break

# print(dex)
# print(los[dex])
