# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob|
import json
import logging
import os
import time

# Import third-party modules
import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker

# Import user-defined modules
import xmask as xm
import xobjects as xo
import xtrack as xt

sns.set_theme(style="ticks")
# %%
collider = xt.Multiline.from_json('/afs/cern.ch/user/a/aradosla/example_DA_study_mine/master_study/scans/example_tunescan/base_collider/collider/collider.json')
data = pd.read_parquet('output_particles_new.parquet')

# %%
line = collider['lhcb1']
tw0 = line.twiss()
betx = tw0.betx[0]
bety = tw0.bety[0]
betx_rel = data.beta0[0]
gamma_rel = data.gamma0[0]
print(betx, bety)
plt.plot(data.x, data.y, '.')


# %%
data[data.y == -0.001033215361951699]

# %%
dfs = []
files = glob.glob('/afs/cern.ch/user/a/aradosla/example_DA_study_mine/master_study/scans/example_tunescan/base_collider/xtrack_*/*.parquet')
for file in files:
    df = pd.read_parquet(file)
    dfs.append(df)
    
# %%
#dfs = [data_0000, data_0001, data_0002, data_0003]
all = pd.concat(dfs)
N_turns = int(max(np.unique(all.at_turn.values)))+1
N_particles = len(all[all.at_turn == 0].x.values)
# %%
print(all)
 # %%
# Check the state of the particles, how many are lost
survived_percent = np.sum(all.state)/len(all) * 100
print(f'Survived particles {survived_percent}%')

lost_particles = 100 - survived_percent
print(f'Lost particles {lost_particles}%')
# %%
def loss_percent(all):
    survived_all = []
    lost_all = []
    for turn in np.unique(all.at_turn.values):
        all_turn = all[all.at_turn == turn]
        survived_percent = np.sum(all_turn.state)/N_particles * 100
        survived_all.append(survived_percent)
        lost_particles = 100 - survived_percent
        lost_all.append(lost_particles)
    return survived_all, lost_all

surv, lost = loss_percent(all)

# %%
plt.plot(surv, '.')
plt.xlabel('Turn number')
plt.ylabel('')
# %%
all
# %%
# Compute emittance
geomx_all_std = []
geomy_all_std = []
normx_all_std = []
normy_all_std = []


for turn in range(int(max(np.unique(all.at_turn.values)))+1):
    sigma_delta = float(np.std(all[all.at_turn == turn].delta))
    sigma_x = float(np.std(all[all.at_turn == turn].x))
    sigma_y = float(np.std(all[all.at_turn == turn].y))
    
    geomx_emittance = (sigma_x**2-(tw0[:,0]["dx"][0]*sigma_delta)**2)/tw0[:,0]["betx"][0]
    normx_emittance = geomx_emittance*(all.gamma0[0]*all.beta0[0])
    geomx_all_std.append(geomx_emittance)
    normx_all_std.append(normx_emittance)

    geomy_emittance = (sigma_y**2-(tw0[:,0]["dy"][0]*sigma_delta)**2)/tw0[:,0]["bety"][0]
    normy_emittance = geomy_emittance*(all.gamma0[0]*all.beta0[0])
    geomy_all_std.append(geomy_emittance)
    normy_all_std.append(normy_emittance)

# %%
# Now normalize with the sqrt of the geom emittance * beta optical
    
x_all = []
y_all = []

for turn in range(int(max(np.unique(all.at_turn.values)))+1):
    x = all[all.at_turn == turn].x / (np.sqrt(geomx_all_std[turn]) * betx)
    y = all[all.at_turn == turn].y / (np.sqrt(geomy_all_std[turn]) * bety)
    x_all.append(x)
    y_all.append(y)


print(len(x))
# %%

all_turns = np.sort(all.at_turn.values)
result_df = pd.DataFrame({'x': pd.concat(x_all), 'y': pd.concat(y_all), 'at_turn': all_turns})
plt.plot(result_df[result_df.at_turn == 0].x, alpha = 0.5)

display(result_df)
result_df.to_parquet('result_df.parquet')
# %%
result_df = pd.read_parquet('result_df.parquet')


# %%
sns.set_theme(style="ticks")
g = sns.JointGrid(data=result_df[result_df.at_turn ==0], x="x", y="y", marginal_ticks=True, space=0.4)

# Set a log scaling on the y axis
#g.ax_joint.set(yscale="log")

# Create an inset legend for the histogram colorbar
cax = g.figure.add_axes([.15, .55, .02, .2])

# Add the joint and marginal histogram plots
g.plot_joint(
    sns.histplot, discrete=(False, False),
    cmap="light:#03012d", cbar=True, cbar_ax = cax
)
g.plot_marginals(sns.histplot, element="step", color="#03012d")
# %%
# Check how many particles are above a certain dynamic aperture
#collider['lhcb1'].twiss()[:,0].betx


#for i in data.y:
#    if abs(i) > 0.001:
        #print(i)
        #print(np.where(data.y == i)[0])
result_df['state'] = 1
survived_all = []
lost_all = []
for turn in range(200):
    sigma = 2
    mean_x = np.mean(result_df[result_df.at_turn == turn].x)
    stdv_x = np.std(result_df[result_df.at_turn == turn].x)
    sigma6_x = mean_x + sigma * stdv_x
    mean_y = np.mean(result_df[result_df.at_turn == turn].y)
    stdv_y = np.std(result_df[result_df.at_turn == turn].y)
    sigma6_y = mean_y + sigma * stdv_y
    condition = (result_df.at_turn == turn) & ((result_df.x > sigma6_x) | (result_df.y > sigma6_y))
    if condition.any():
        #print(turn)
        # Update specific rows with '-1' for 'state' column
        result_df.loc[(result_df.at_turn >= turn) & condition, 'x'] = 0
        result_df.loc[(result_df.at_turn >= turn) & condition, 'y'] = 0
        result_df.loc[(result_df.at_turn >= turn) & condition, 'state'] = -1
    survived_percent = np.sum(result_df[result_df.at_turn == turn].state)/N_particles * 100
    survived_all.append(survived_percent)
    lost_particles = 100 - survived_percent
    lost_all.append(lost_particles)


#surv_part, lost_part = loss_percent(result_df)
plt.plot(survived_all, label='particles')
plt.plot(lost_all, label='losses')

plt.legend()






# %% 
'''

x_data = all.x
y_data = all.y
px_data = all.px
py_data = all.py
zeta_data = all.zeta
pz_data = all.ptau
delta_data = all.delta
x = x_data.T
y = y_data.T
px = px_data.T
py = py_data.T
zeta = zeta_data.T
pzeta = pz_data.T
N_particles = int(len(data)/(max(np.unique(data.at_turn)) + 1))
N_turns = 3

Jx = np.zeros((N_turns, int(N_particles)))
Jy = np.zeros((N_turns, int(N_particles))) 
errorx = np.zeros(N_turns)
errory = np.zeros(N_turns)

betx_rel =data.beta0[0]
gamma_rel = data.gamma0[0]
W = line.twiss()['W_matrix'][0]

W_inv = np.linalg.inv(W)
tw_full_inverse = line.twiss(use_full_inverse=True)['W_matrix'][0]

n_repetitions = N_turns
n_particles = N_particles

inv_w = W_inv

phys_coord = np.array([x.values,px.values,y.values,py.values,zeta.values,pzeta.values])
phys_coord = phys_coord.astype(float)
phys_coord[phys_coord==0.]=np.nan
# %%

norm_coord = np.zeros_like(phys_coord)
for i in range(n_repetitions):
    norm_coord[:,i,:] = np.matmul(inv_w, (phys_coord[:,i,:]))

for i in range(N_turns):
    Jx[i,:] = (pow(norm_coord[0, i, :],2)+pow(norm_coord[1, i, :],2))/2 
    print(Jx[i,:])
    Jy[i,:] = (pow(norm_coord[2, i, :],2)+pow(norm_coord[3, i, :],2))/2 

emitx = np.nanmean(Jx, axis=1)*(betx_rel*gamma_rel)
emity = np.nanmean(Jy, axis=1)*(betx_rel*gamma_rel)
'''
