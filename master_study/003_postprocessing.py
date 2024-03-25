# %%

# ==================================================================================================
# --- Imports
# ==================================================================================================
import logging
import time

import pandas as pd
import tree_maker
import yaml

# ==================================================================================================
# --- Load tree of jobs
# ==================================================================================================

# Start of the script
print("Analysis of output simulation files started")
start = time.time()

# Load Data
study_name = "example_tunescan"
fix = "/scans/" + study_name
root = tree_maker.tree_from_json(fix[1:] + "/tree_maker.json")
# Add suffix to the root node path to handle scans that are not in the root directory
root.add_suffix(suffix=fix)


# ==================================================================================================
# --- # Browse simulations folder and extract relevant observables
# ==================================================================================================
l_problematic_sim = []
l_df_to_merge = []
for node in root.generation(1):
    with open(f"{node.get_abs_path()}/config.yaml", "r") as fid:
        config_parent = yaml.safe_load(fid)
    for node_child in node.children:
        with open(f"{node_child.get_abs_path()}/config.yaml", "r") as fid:
            config_child = yaml.safe_load(fid)

        try:
            # Read the particle path as relative
            try:
                particle = pd.read_parquet(
                    f"{node_child.get_abs_path()}/{config_child['config_simulation']['particle_file']}"
                )

            # If it doesn't work, try to read it as absolute
            except:
                particle = pd.read_parquet(f"{config_child['config_simulation']['particle_file']}")

            df_sim = pd.read_parquet(f"{node_child.get_abs_path()}/output_particles_new.parquet")

        except Exception as e:
            print(e)
            l_problematic_sim.append(node_child.get_abs_path())
            continue

        # Register paths and names of the nodes
        df_sim["path base collider"] = f"{node.get_abs_path()}"
        df_sim["name base collider"] = f"{node.name}"
        df_sim["path simulation"] = f"{node_child.get_abs_path()}"
        df_sim["name simulation"] = f"{node_child.name}"

        # Get node parameters as dictionnaries for parameter assignation
        dic_child_collider = node_child.parameters["config_collider"]
        dic_child_simulation = node_child.parameters["config_simulation"]
        try:
            dic_parent_collider = node.parameters["config_mad"]
        except:
            print("No parent collider could be loaded")
        dic_parent_particles = node.parameters["config_particles"]

        # Get which beam is being tracked
        df_sim["beam"] = dic_child_simulation["beam"]

        # Get scanned parameters (complete with the requested scanned parameters)
        df_sim["qx"] = dic_child_collider["config_knobs_and_tuning"]["qx"]["lhcb1"]
        df_sim["qy"] = dic_child_collider["config_knobs_and_tuning"]["qy"]["lhcb1"]
        df_sim["dqx"] = dic_child_collider["config_knobs_and_tuning"]["dqx"]["lhcb1"]
        df_sim["dqy"] = dic_child_collider["config_knobs_and_tuning"]["dqy"]["lhcb1"]
        df_sim["i_bunch_b1"] = dic_child_collider["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b1"
        ]
        df_sim["i_bunch_b2"] = dic_child_collider["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b2"
        ]
        df_sim["num_particles_per_bunch"] = dic_child_collider["config_beambeam"][
            "num_particles_per_bunch"
        ]
        df_sim["i_oct_b1"] = dic_child_collider["config_knobs_and_tuning"]["knob_settings"][
            "i_oct_b1"
        ]
        df_sim["i_oct_b2"] = dic_child_collider["config_knobs_and_tuning"]["knob_settings"][
            "i_oct_b2"
        ]
        df_sim["crossing_angle"] = abs(
            float(dic_child_collider["config_knobs_and_tuning"]["knob_settings"]["on_x1"])
        )

        # Merge with particle data
        df_sim_with_particle = pd.merge(df_sim, particle, on=["particle_id"])
        l_df_to_merge.append(df_sim_with_particle)

# ==================================================================================================
# --- # Merge all jobs outputs in one dataframe and save it
# ==================================================================================================
# %%
# Merge the dataframes from all simulations together
df_all_sim = pd.concat(l_df_to_merge)

# Extract the particles that were lost for DA computation
df_lost_particles = df_all_sim[df_all_sim["state"] != 1]  # Lost particles

# Check if the dataframe is empty
if df_lost_particles.empty:
    print("No unstable particles found, the output dataframe will be empty.")
my_final = df_all_sim
my_final.to_parquet(f"scans/{study_name}/da.parquet")
print('Particle_id', df_all_sim['particle_id'])
print("Final dataframe for current set of simulations: ", my_final)

end = time.time()

print("Elapsed time: ", end - start)
quit()
# %%
# Group by working point (Update this with the knobs you want to group by !)
group_by_parameters = ["name base collider", "qx", "qy"]
# We always want to keep beam in the final result
group_by_parameters = ["beam"] + group_by_parameters
l_parameters_to_keep = [
    "normalized amplitude in xy-plane",
    "qx",
    "qy",
    "dqx",
    "dqy",
    "i_bunch_b1",
    "i_bunch_b2",
    "num_particles_per_bunch",
    "crossing_angle",
]

# Min is computed in the groupby function, but values should be identical
my_final = pd.DataFrame(
    [
        df_lost_particles.groupby(group_by_parameters)[parameter].min()
        for parameter in l_parameters_to_keep
    ]
).transpose()

# Save data and print time
my_final.to_parquet(f"scans/{study_name}/da.parquet")
print("Final dataframe for current set of simulations: ", my_final)
end = time.time()
print("Elapsed time: ", end - start)

# %%
