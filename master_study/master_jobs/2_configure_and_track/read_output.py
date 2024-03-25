# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
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
import matplotlib.ticker as ticker
sns.set_theme(style="ticks")
# %%
# ========================================================================================================
# Loading the collider used in the simulation

#collider = xt.Multiline.from_json('/afs/cern.ch/user/a/aradosla/example_DA_study_mine/master_study/scans/example_tunescan/base_collider/collider/collider.json')
collider = xt.Multiline.from_json('collider.json')
data = pd.read_parquet('output_particles_new.parquet')


# %%
# ========================================================================================================
# Twiss parameters used for the analysis

line = collider['lhcb1']
tw0 = line.twiss()
betx = tw0.betx[0]
bety = tw0.bety[0]
alfx = tw0.alfx[0]
gamx = tw0.gamx[0]
betx_rel = data.beta0[0]
gamma_rel = data.gamma0[0]
print(betx, bety)
plt.plot(data.x, data.y, '.')

# %%
#data[data.y == -0.001033215361951699]
dir(tw0)
# %%
# ========================================================================================================
# Loading the results and introduce a particle object again for the line.get_normalized_coordinates()

dfs = []
file_names = []
#files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim-62.31-60.32/xtrack_*/*.parquet')
files = glob.glob('/eos/user/a/aradosla/SWAN_projects/Noise_sim/xtrack_*/*.parquet')
for file in files:
    try:
        df = pd.read_parquet(file)
        print('read')
        dfs.append(df)
        path = file
        head, tail = os.path.split(path)
        parent_folder, current_folder = os.path.split(head)
        particles = collider['lhcb1'].build_particles(
            x=df.x.values,
            y=df.y.values,
            px = df.px.values,
            py = df.py.values,
            zeta = df.zeta.values,
            delta=df.delta.values,
        )

        file_names.append([os.path.join(current_folder, tail)]*len(df.x))
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        continue  # Skip to the next file if an error occurs

        

file_names = np.array(file_names).flatten()
first_digit = str(len(file_names))[0]
#betx_rel = df.beta0[0]
#gamma_rel = df.gamma0[0]
# %%
# ========================================================================================================
# Concatenate the files and extract the number of turns and particles

#dfs = [data_0000, data_0001, data_0002, data_0003]
all = pd.concat(dfs, axis = 0)
all['file_name'] = file_names
N_turns = int(max(np.unique(all.at_turn.values)))+1
N_particles = len(all[all.at_turn == 0].x.values)
# %%
print(all)
 # %%
# ========================================================================================================
# Check the state of the particles, how many are lost just a try

survived_percent = np.sum(all.state)/len(all) * 100
print(f'Survived particles {survived_percent}%')

lost_particles = 100 - survived_percent
print(f'Lost particles {lost_particles}%')
# %%
# ========================================================================================================
# Loss function, the same as the previous cell, but as a function

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
# ========================================================================================================
# Compute statistical emittance, not needed here!!!

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
# ========================================================================================================
# Now normalize with the sqrt of the geom emittance * beta optical
    
x_all = []
y_all = []
px_all = []
py_all = []
zeta_all = []
delta_all = []
norm_emit = 2.2e-6

"""
for turn in range(int(max(np.un0
"""
for turn in range(int(max(np.unique(all.at_turn.values)))+1):
    x = all[all.at_turn == turn].x #/ (np.sqrt(geomx_all_std[turn]) * betx)
    y = all[all.at_turn == turn].y #/ (np.sqrt(geomy_all_std[turn]) * bety)       # there should be the gamma too
    px = all[all.at_turn == turn].px #/ (np.sqrt(geomy_all_std[turn]) * bety)     # I think here we divide by beta
    py = all[all.at_turn == turn].py
    zeta = all[all.at_turn == turn].zeta
    delta = all[all.at_turn == turn].delta
    x_all.append(x)
    y_all.append(y)
    px_all.append(px)
    py_all.append(py)
    zeta_all.append(zeta)
    delta_all.append(delta)
#pzeta_all = np.zeros_like(zeta_all)




print(len(x))
 # %%
# ========================================================================================================
# Normalised coordinates using the Wigner rotation matrix

#all_turns = np.sort(all.at_turn.values)
all_turns_num = np.arange(N_turns)
all_turns_one = np.repeat(all_turns_num, N_particles/float(first_digit))
all_turns = np.tile(all_turns_one, int(first_digit))
result_df = pd.DataFrame({'x': pd.concat(x_all), 'y': pd.concat(y_all), 'px': pd.concat(px_all), 'at_turn': all_turns, 'file': file_names})
plt.plot(result_df[result_df.at_turn == 0].x, alpha = 0.5)

W = tw0['W_matrix'][0]

W_inv = np.linalg.inv(W)
tw_full_inverse = line.twiss(use_full_inverse=True)['W_matrix'][0]


inv_w = W_inv

phys_coord = np.array([x_all, px_all, y_all, py_all, zeta_all, delta_all])
phys_coord = phys_coord.astype(float) 
#phys_coord[phys_coord==0.]=np.nan
norm_coord = np.zeros_like(phys_coord)            # I don't think that is correct, I'm running the simulation again 
for i in range(N_turns):
    norm_coord[:,i,:] = np.matmul(inv_w, (phys_coord[:,i,:]))

#Jx = np.zeros((1, N_particles))
#Jx[0,:] = (pow(norm_coord[0, 0, :],2)+pow(norm_coord[1, 0, :],2))/2 

#result_df_norm = pd.DataFrame({'x': pd.concat(x_all), 'y': pd.concat(y_all), 'px': pd.concat(px_all), 'at_turn': all_turns, 'file': file_names})


# %%
# ========================================================================================================
# Using the built in function to see if the normalised coordinates I compute are matching

coord_norm_func = tw0.get_normalized_coordinates(particles, nemitt_x = 2.2e-6, nemitt_y = 2.2e-6)
coord_norm_func.show()

plt.plot(coord_norm_func.x_norm[:3000], coord_norm_func.px_norm[:3000], '.')
plt.xlabel('x')

# %% 

particle_list = [
    (particle_id, x, y, px, py, zeta, delta)
    for particle_id, (x, y, px, py, zeta, delta) in enumerate(zip(coord_norm_func.x_norm[:30000], coord_norm_func.y_norm[:3000], coord_norm_func.px_norm[:3000], coord_norm_func.py_norm[:3000], coord_norm_func.zeta_norm[:3000], coord_norm_func.pzeta_norm[:3000]))]
print('first',particle_list)
# Split distribution into several chunks for parallelization

particle_list = np.array(particle_list)
array_of_lists = np.array([arr.tolist() for arr in particle_list])
particle_list = array_of_lists
print('second',particle_list)



coord = pd.DataFrame(
    particle_list,
    columns=["particle_id", "x", "y", "px", "py", "zeta", "pzeta"]
)
# %%
# ========================================================================================================
sigmax_col = np.sqrt(norm_emit / gamma_rel * betx)  # geom_emit = norm_emit / gamma_rel
aperture = np.arange(2, 0, -0.01)
survived = []
current_turn = 0
for current_aperture in aperture:
    current_df = coord
    #current_df = coord[coord["turns"] == current_turn]
    initial_number_of_particles = len(current_df)
    current_df = current_df[(current_df["x"]**2 + (current_df["px"])**2) * 100 < (current_aperture)**2]
    survived.append(100 - len(current_df)/initial_number_of_particles*100.)
fig, ax = plt.subplots()
plt.plot(aperture, survived, lw=4, label = 'Suvived')
plt.plot(aperture, [np.exp(-x**2/2.0)*100. for x in aperture], c='r', linestyle='--', lw=2)
plt.xlabel('Sigma')

plt.ylabel('Survived particles %')


sigmax_col = np.sqrt(2.2e-6 / gamma_rel * betx) # %%

    
# %%
row_names = ['x', 'y', 'px', 'py', 'zeta', 'pzeta']
flat_norm = []
for i in norm_coord:
    flat_norm.append(i.flatten())

data_dict = {}
for i, name in enumerate(row_names):
    data_dict[name] = flat_norm[i]

data_dict['turns'] = all_turns
data_dict['file_names'] = file_names
df = pd.DataFrame(data_dict)

# %% 

plt.figure(figsize = (20, 9))
plt.subplot(1, 2, 1)
plt.plot(result_df.x[:30000]/np.sqrt(norm_emit / gamma_rel * betx), result_df.px[:30000]/(np.sqrt(norm_emit/gamma_rel/betx)), '.')
plt.xlabel('x', size = 20)
plt.ylabel('px', size = 20)
plt.title('Physical coordinates', size = 25)

# Set the x-axis tick formatter
formatter = ticker.FuncFormatter(lambda x, _: '{:g}'.format(x))
plt.gca().xaxis.set_major_formatter(formatter)
plt.subplot(1, 2, 2)
plt.plot(df.x[0:30000]/np.sqrt(norm_emit), df.px[:30000]/np.sqrt(norm_emit), '.')
plt.xlabel('x', size=20)
plt.ylabel('px', size = 20)
plt.title('Normalised coordinates', size = 25)

plt.subplots_adjust(hspace=0.5)  # Adjust the vertical space

# %%

# scraping

sigmax_col = np.sqrt(norm_emit / gamma_rel * betx)  # geom_emit = norm_emit / gamma_rel
aperture = np.arange(0, 6, 0.01)
survived = []
current_turn = 0
for current_aperture in aperture:
    current_df = df[df["turns"] == current_turn]
    initial_number_of_particles = len(current_df)
    current_df = current_df[(current_df["x"]**2 + (current_df["px"])**2) < (current_aperture)**2]
    survived.append(100 - len(current_df)/initial_number_of_particles*100.)
fig, ax = plt.subplots()
plt.plot(aperture, survived, lw=4, label = 'Suvived')
plt.plot(aperture, [np.exp(-x**2/2.0)*100. for x in aperture], c='r', linestyle='--', lw=2)
plt.xlabel('Sigma')

plt.ylabel('Survived particles %')


sigmax_col = np.sqrt(2.2e-6 / gamma_rel * betx) # %%


result_df.to_parquet('/eos/user/a/aradosla/SWAN_projects/Noise_sim/result_df.parquet')




# %%

# Plot seaborn, distibution in x and y 
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

### ----------------------------------- New scrapping ----------------------------------- ###
sigmax_col = np.sqrt(2.2e-6 / gamma_rel * betx)
sigmay_col = np.sqrt(2.2e-6 / gamma_rel * bety)

#sigmax_all = [6*sigmax_col, 5*sigmax_col, 4*sigmax_col, 3*sigmax_col, 2*sigmax_col, 1*sigmax_col]
#sigmay_all = [6*sigmay_col, 5*sigmay_col, 4*sigmay_col, 3*sigmay_col, 2*sigmay_col, 1*sigmay_col]
sigmax_all = np.arange(10, 0, -0.01)*sigmax_col
sigmay_all = np.arange(10, 0, -0.01)*sigmay_col

result_df = pd.read_parquet('result_df.parquet')
result_df['state'] = 1
survived_all = []
lost_all = []
'''
for sigma in range(len(sigmax_all)):
    result_df_copy = result_df.copy()
    condition = ((abs(result_df_copy[result_df_copy.at_turn == 0].x) > sigmax_all[sigma]) | (abs(result_df_copy[result_df_copy.at_turn == 0].y) > sigmay_all[sigma]))
    if condition.any():
        part = result_df_copy[(result_df_copy.at_turn == 0)]
        result_df_copy[(result_df_copy.at_turn == 0)].loc[condition.values, 'x'] = 0
        result_df_copy[(result_df_copy.at_turn == 0)].loc[condition.values, 'y'] = 0
        result_df_copy[(result_df_copy.at_turn == 0)].loc[condition.values, 'state'] = -1
    survived_percent = np.sum(result_df_copy[(result_df_copy.at_turn == 0)].state)/len(result_df_copy[(result_df_copy.at_turn == 0)].state) * 100
    survived_all.append(survived_percent)
    lost_particles = 100 - survived_percent
    lost_all.append(lost_particles)

plt.plot(np.arange(0, 10, 0.01), np.array(survived_all)/2+50)
plt.xlabel('Sigma')
plt.ylabel('Survived particles %')
'''
# %%
# scraping
aperture = np.arange(0, 6, 0.1)
survived = []
current_turn = 0
for current_aperture in aperture:
    current_df = result_df[result_df["at_turn"] == current_turn]
    initial_number_of_particles = len(current_df)
    current_df = current_df[(gamx*(current_df["x"])**2 + 2*alfx*(current_df["x"])*(current_df["px"]) + betx*(current_df["px"])**2)  < (current_aperture*sigmax_col)**2]
    survived.append(100 - len(current_df)/initial_number_of_particles*100.)
fig, ax = plt.subplots()
plt.plot(aperture, survived, lw=4, label = 'Suvived')
plt.plot(aperture, [np.exp(-x**2/2.0)*100. for x in aperture], c='r', linestyle='--', lw=2)
plt.xlabel('Sigma')

plt.ylabel('Survived particles %')


# %%
# Calculate sigmax and sigmay

sigmax_col = np.sqrt(2.2e-6 / gamma_rel * betx)
sigmay_col = np.sqrt(2.2e-6 / gamma_rel * bety)

# Generate sigmax and sigmay arrays
sigmax_all = np.arange(10, 0, -0.01) * sigmax_col
sigmay_all = np.arange(10, 0, -0.01) * sigmay_col

# Read the DataFrame
result_df = pd.read_parquet('result_df.parquet')
result_df['state'] = 1

# Initialize lists to store survival percentages
survived_all = []
lost_all = []

# Loop over different sigma values
for sigma in range(len(sigmax_all)):
    # Define condition based on sigma values
    condition = ((abs(result_df['x']) > sigmax_all[sigma]) | (abs(result_df['y']) > sigmay_all[sigma]))
    
    # Update DataFrame based on condition
    result_df.loc[condition & (result_df['at_turn'] == 0), 'x'] = 0
    result_df.loc[condition & (result_df['at_turn'] == 0), 'y'] = 0
    result_df.loc[condition & (result_df['at_turn'] == 0), 'state'] = -1
    
    # Calculate survival percentage
    survived_percent = np.sum(result_df['state'] == 1) / len(result_df) * 100
    survived_all.append(survived_percent)
    lost_particles = 100 - survived_percent
    lost_all.append(lost_particles)

# Plot the results
plt.plot(np.arange(0, 10, 0.01), np.array(survived_all) / 2 + 50)
plt.xlabel('Sigma')
plt.ylabel('Survived particles %')
plt.show()

# %%


### ----------------------------------- New scrapping ----------------------------------- ###
sigmax_col = np.sqrt(2.2e-6 / gamma_rel * betx)
sigmay_col = np.sqrt(2.2e-6 / gamma_rel * bety)

#sigmax_all = [6*sigmax_col, 5*sigmax_col, 4*sigmax_col, 3*sigmax_col, 2*sigmax_col, 1*sigmax_col]
#sigmay_all = [6*sigmay_col, 5*sigmay_col, 4*sigmay_col, 3*sigmay_col, 2*sigmay_col, 1*sigmay_col]
sigmax_all = np.arange(10, 0, -0.01)*sigmax_col
sigmay_all = np.arange(10, 0, -0.01)*sigmay_col

result_df = pd.read_parquet('result_df.parquet')
result_df['state'] = 1
survived_all = []
lost_all = []

turn_max = max(np.unique(result_df.at_turn))
for sigma in range(len(sigmax_all)):
    result_df_copy = result_df.copy()
    #result_df_copy = data.copy()
    condition = ((abs(result_df_copy[result_df_copy.at_turn == turn_max].x) > sigmax_all[sigma]) | (abs(result_df_copy[result_df_copy.at_turn == turn_max].y) > sigmay_all[sigma]))
    if condition.any():
        part = result_df_copy[(result_df_copy.at_turn == turn_max)]
        part.loc[condition.values, 'x'] = 0
        part.loc[condition.values, 'y'] = 0
        part.loc[condition.values, 'state'] = -1
    survived_percent = np.sum(part.state)/len(part.state) * 100
    survived_all.append(survived_percent)
    lost_particles = 100 - survived_percent
    lost_all.append(lost_particles)

plt.plot(np.arange(0, 10, 0.01), np.array(survived_all)/2+50)
plt.xlabel('Sigma')
plt.ylabel('Survived particles %')

# %%

sigmax_col = np.sqrt(2.2e-6 / gamma_rel * betx)
sigmay_col = np.sqrt(2.2e-6 / gamma_rel * bety)

sigmax_all = [6*sigmax_col, 5*sigmax_col, 4*sigmax_col, 3*sigmax_col, 2*sigmax_col, 1*sigmax_col]
sigmay_all = [6*sigmay_col, 5*sigmay_col, 4*sigmay_col, 3*sigmay_col, 2*sigmay_col, 1*sigmay_col]
#data['state'] = 1
survived_all = []
lost_all = []
for sigma in range(len(sigmax_all)):
    result_df_copy = result_df.copy()
    condition = ((abs(result_df_copy[result_df_copy.at_turn == max(np.unique(result_df_copy.at_turn))].x) > sigmax_all[sigma]) | (abs(result_df_copy[result_df_copy.at_turn == max(np.unique(result_df_copy.at_turn))].y) > sigmay_all[sigma]))
    if condition.any():
        result_df_copy.loc[(result_df_copy.at_turn == max(np.unique(result_df_copy.at_turn))) & condition, 'x'] = 0
        result_df_copy.loc[(result_df_copy.at_turn == max(np.unique(result_df_copy.at_turn))) & condition, 'y'] = 0
        result_df_copy.loc[(result_df_copy.at_turn == max(np.unique(result_df_copy.at_turn))) & condition, 'state'] = -1
    survived_percent = np.sum(result_df_copy[result_df_copy.at_turn == max(np.unique(result_df_copy.at_turn))].state)/len(result_df_copy[result_df_copy.at_turn == max(np.unique(result_df_copy.at_turn))].state) * 100
    survived_all.append(survived_percent)
    lost_particles = 100 - survived_percent
    lost_all.append(lost_particles)

plt.plot(np.arange(len(survived_all)), survived_all)


# %%
# Scrapping

result_df = pd.read_parquet('result_df.parquet')
result_df['state'] = 1
survived_all = []
lost_all = []
for turn in range(10):
    sigma = 2.3
    mean_x = np.mean(result_df[result_df.at_turn == turn].x)
    stdv_x = np.std(result_df[result_df.at_turn == turn].x)
    sigma_x = mean_x + sigma * stdv_x
    mean_y = np.mean(result_df[result_df.at_turn == turn].y)
    stdv_y = np.std(result_df[result_df.at_turn == turn].y)
    sigma_y = mean_y + sigma * stdv_y
    condition = ((abs(result_df.x) > sigma_x) | (abs(result_df.y) > sigma_y))
    if condition.any():
        #print(turn)
        # Update specific rows with '-1' for 'state' column
        result_df.loc[(result_df.at_turn >= turn) & condition, 'x'] = 0
        print(result_df.loc[(result_df.at_turn >= turn) & condition, 'x'])
        result_df.loc[(result_df.at_turn >= turn) & condition, 'y'] = 0
        result_df.loc[(result_df.at_turn >= turn) & condition, 'state'] = -1
    survived_percent = np.sum(result_df[result_df.at_turn == turn].state)/N_particles * 100
    survived_all.append(survived_percent)
    lost_particles = 100 - survived_percent
    lost_all.append(lost_particles)


print(lost_all)
#surv_part, lost_part = loss_percent(result_df)
plt.plot(np.array(survived_all), label='particles')
plt.plot(np.array(lost_all), label='losses')
plt.xlabel('Turns')
plt.ylabel('Intensity %')
plt.legend()



result_df = pd.read_parquet('result_df.parquet')
result_df['state'] = 1
survived_all = []
lost_all = []

for turn in range(200):
    sigma = 2

    mean_x = np.mean(result_df[result_df.at_turn == turn].x)
    stdv_x = np.std(result_df[result_df.at_turn == turn].x)
    sigma_x = mean_x + sigma * stdv_x
    mean_y = np.mean(result_df[result_df.at_turn == turn].y)
    stdv_y = np.std(result_df[result_df.at_turn == turn].y)
    sigma_y = mean_y + sigma * stdv_y
    
    condition = (result_df.at_turn == turn) & ((abs(result_df.x) > sigma_x) | (abs(result_df.y) > sigma_y))
    print(len(condition))
    if condition.any():
        # Update specific rows with '-1' for 'state' column for all subsequent turns
        for subsequent_turn in range(turn, 200):
            condition_subsequent_turn = (result_df.at_turn == subsequent_turn) & condition
            print(len(condition_subsequent_turn))
            result_df.loc[condition_subsequent_turn, 'x'] = 0
            result_df.loc[condition_subsequent_turn, 'y'] = 0
            result_df.loc[condition_subsequent_turn, 'state'] = -1
    
    survived_percent = np.sum(result_df[result_df.at_turn == turn].state) / N_particles * 100
    survived_all.append(survived_percent)
    lost_particles = 100 - survived_percent
    lost_all.append(lost_particles)
    
plt.plot(np.array(survived_all), label='particles')
plt.plot(np.array(lost_all), label='losses')
plt.xlabel('Turns')
plt.ylabel('Intensity %')
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
