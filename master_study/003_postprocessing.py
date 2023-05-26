# ==================================================================================================
# --- Imports
# ==================================================================================================
import tree_maker
import yaml
import pandas as pd
import time
import logging

# ==================================================================================================
# --- Load tree of jobs
# ==================================================================================================

# Start of the script
print("Analysis of output simulation files started")
start = time.time()

# Load Data
study_name = "opt_flathv_75_1500_withBB_chroma5_1p4_eol"  # "example_HL_tunescan"
fix = "/scans/" + study_name
root = tree_maker.tree_from_json(fix[1:] + "/tree_maker_" + study_name + ".json")
# Add suffix to the root node path to handle scans that are not in the root directory
root.add_suffix(suffix=fix)


# ==================================================================================================
# --- Define a function to get the collider parameters associated with a given simulation
# ==================================================================================================


# Function for parameter assignation
def assign_parameter(parameter, group, df_sim, dic_child, dic_parent=None):
    if group in dic_child:
        if parameter in dic_child[group]:
            df_sim[parameter] = dic_child[group][parameter]
            return df_sim

    # If the parameters have not been scanned, they must be found in the base collider configuation
    if dic_parent is not None:
        conf_knobs_and_tuning = dic_parent["config_knobs_and_tuning"]
        config_bb = dic_parent["config_beambeam"]

        # Group 1 contains separation (on_sep2, on_sep8h, on_sep8v) and bunch intensity,
        # update with other knobs if needed
        if group == "group_1":
            print(conf_knobs_and_tuning["knob_settings"])
            if parameter in conf_knobs_and_tuning["knob_settings"]:
                df_sim[parameter] = conf_knobs_and_tuning["knob_settings"][parameter]
            elif parameter in config_bb:
                df_sim[parameter] = config_bb[parameter]
            else:
                logging.warning(
                    f"The parameter {parameter} is assumed to be a knob belonging to"
                    " conf_knobs_and_tuning['knob_settings'] or config_bb. Please update script"
                    " accordingly."
                )

        # Group 2 contains crossing-angle, chromaticity, tune and octupoles,
        # update with other knobs if needed
        elif group == "group_2":
            if parameter in conf_knobs_and_tuning["knob_settings"]:
                df_sim[parameter] = conf_knobs_and_tuning["knob_settings"][parameter]
            elif parameter in conf_knobs_and_tuning:
                if (
                    "lhcb1" in conf_knobs_and_tuning[parameter]
                    or "lhcb2" in conf_knobs_and_tuning[parameter]
                ):
                    # chromaticity and tune are handled only for 1 beam... Update this if required
                    df_sim[parameter] = conf_knobs_and_tuning[parameter]["lhcb1"]
                else:
                    df_sim[parameter] = conf_knobs_and_tuning[parameter]
            else:
                logging.warning(
                    f"The parameter {parameter} is assumed to be a knob belonging to"
                    " conf_knobs_and_tuning['knob_settings'] or conf_knobs_and_tuning. Please"
                    " update script accordingly."
                )

        # Group 3 contains bunch number, update with other parameters if needed
        elif group == "group_3":
            if parameter in config_bb["mask_with_filling_pattern"]:
                df_sim[parameter] = config_bb["mask_with_filling_pattern"][parameter]
            else:
                logging.warning(
                    f"The parameter {parameter} is assumed to be a knob belonging to"
                    " config_bb['mask_with_filling_pattern']. Please update script accordingly."
                )
    return df_sim


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
            particle = pd.read_parquet(
                f"{node_child.get_abs_path()}/{config_child['particle_file']}"
            )
            df_sim = pd.read_parquet(f"{node_child.get_abs_path()}/output_particles.parquet")

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
        dic_parent_collider = node.parameters["config_collider"]
        dic_parent_particles = node.parameters["config_particles"]

        # Get scanned parameters (complete with the scanned parameters)
        df_sim["qx"] = dic_child_collider["config_knobs_and_tuning"]["qx"]["lhcb1"]
        df_sim["qy"] = dic_child_collider["config_knobs_and_tuning"]["qy"]["lhcb1"]
        df_sim["i_bunch_b1"] = dic_child_collider["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b1"
        ]
        df_sim["i_bunch_b2"] = dic_child_collider["config_beambeam"]["mask_with_filling_pattern"][
            "i_bunch_b2"
        ]

        # Merge with particle data
        df_sim_with_particle = pd.merge(df_sim, particle, on=["particle_id"])
        l_df_to_merge.append(df_sim_with_particle)

# ==================================================================================================
# --- # Merge all jobs outputs in one dataframe and save it
# ==================================================================================================

# Merge the dataframes from all simulations together
df_all_sim = pd.concat(l_df_to_merge)

# Extract the particles that were lost for DA computation
df_lost_particles = df_all_sim[df_all_sim["state"] != 1]  # Lost particles

# Check if the dataframe is empty
if df_lost_particles.empty:
    print("No unstable particles found, the output dataframe will be empty.")

# Groupe by working point (Update this with the knobs you want to group by !)
# Median is computed in the groupby function, but values are assumed identical
groupby = ["qx", "qy"]  # ["i_bunch_b1", "i_bunch_b2"]  #
my_final = pd.DataFrame(
    [
        df_lost_particles.groupby(groupby)["normalized amplitude in xy-plane"].min(),
        df_lost_particles.groupby(groupby)["qx"].median(),
        df_lost_particles.groupby(groupby)["qy"].median(),
        df_lost_particles.groupby(groupby)["i_bunch_b1"].median(),
        df_lost_particles.groupby(groupby)["i_bunch_b2"].median(),
    ]
).transpose()

print(my_final)

# Save data and print time
my_final.to_parquet(f"scans/{study_name}/da.parquet")
print(my_final)
end = time.time()
print(end - start)
