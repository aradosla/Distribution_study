
# %%
# ==================================================================================================
# --- Imports
# ==================================================================================================
import copy
import itertools
import json
import os
import shutil
import time

import numpy as np
import yaml
from tree_maker import initialize
from user_defined_functions import (
    generate_run_sh,
    generate_run_sh_htc,
    get_worst_bunch,
    reformat_filling_scheme_from_lpc_alt,
)
# %%
# ==================================================================================================
# --- Initial particle distribution parameters (generation 1)
#
# Below, the user defines the parameters for the initial particles distribution.
# Path for the particle distribution configuration:
# master_study/master_jobs/1_build_distr_and_collider/config.yaml [field config_particles]
# ==================================================================================================

# Define dictionary for the initial particle distribution
d_config_particles = {}

# Radius of the initial particle distribution
d_config_particles["r_min"] = 2
d_config_particles["r_max"] = 10
d_config_particles["n_r"] = 2 * 16 * (d_config_particles["r_max"] - d_config_particles["r_min"])

# Number of angles for the initial particle distribution
d_config_particles["n_angles"] = 5

# Number of split for parallelization
d_config_particles["n_split"] = 5

# ==================================================================================================
# --- Optics collider parameters (generation 1)
#
# Below, the user defines the optics collider parameters. These parameters cannot be scanned.
# Path for the collider configuration:
# master_study/master_jobs/1_build_distr_and_collider/config.yaml [field config_mad]
# ==================================================================================================

### Mad configuration

# Define dictionary for the Mad configuration
d_config_mad = {"beam_config": {"lhcb1": {}, "lhcb2": {}}, "links": {}}

# Optic file path (version, and round or flat)

### For run III
d_config_mad["links"]["acc-models-lhc"] = "/afs/cern.ch/eng/lhc/optics/runIII"
#d_config_mad["optics_file"] = "acc-models-lhc/RunIII_dev/Proton_2024/V0/opticsfile.40"
d_config_mad["optics_file"] = "/afs/cern.ch/eng/lhc/optics/runIII/RunIII_dev/Proton_2023/opticsfile.23"
d_config_mad["ver_hllhc_optics"] = None
d_config_mad["ver_lhc_run"] = 3.0


# Beam energy (for both beams)
beam_energy_tot = 6800
d_config_mad["beam_config"]["lhcb1"]["beam_energy_tot"] = beam_energy_tot
d_config_mad["beam_config"]["lhcb2"]["beam_energy_tot"] = beam_energy_tot


# ==================================================================================================
# --- Base collider parameters (generation 2)
#
# Below, the user defines the standard collider parameters. Some of the values defined here are
# later updated according to the grid-search being done.
# Path for the collider config:
# master_study/master_jobs/2_configure_and_track/config.yaml [field config_collider]
# ==================================================================================================

### Tune and chroma configuration

# Define dictionnary for tune and chroma
d_config_tune_and_chroma = {
    "qx": {},
    "qy": {},
    "dqx": {},
    "dqy": {},
}
for beam in ["lhcb1", "lhcb2"]:
    d_config_tune_and_chroma["qx"][beam] = 62.31
    d_config_tune_and_chroma["qy"][beam] = 60.32
    d_config_tune_and_chroma["dqx"][beam] = 15.0
    d_config_tune_and_chroma["dqy"][beam] = 15.0

# Value to be added to linear coupling knobs
d_config_tune_and_chroma["delta_cmr"] = 0.001
d_config_tune_and_chroma["delta_cmi"] = 0.0

### Knobs configuration

# Define dictionary for the knobs settings
d_config_knobs = {}

# Exp. configuration in IR1, IR2, IR5 and IR8
d_config_knobs["on_x1"] = -145.000
d_config_knobs["on_sep1"] = 0.0
d_config_knobs["phi_IR1"] = 180.000

d_config_knobs["on_x2h"] = 0.000
d_config_knobs["on_sep2h"] = 1.0  # 1.000
d_config_knobs["on_x2v"] = 200.000
d_config_knobs["on_sep2v"] = 0.000
d_config_knobs["phi_IR2"] = 90.000

d_config_knobs["on_x5"] = 145.000
d_config_knobs["on_sep5"] = 0.0
d_config_knobs["phi_IR5"] = 90.000

d_config_knobs["on_x8h"] = 0.000
d_config_knobs["on_sep8h"] = -0.01  # -1.000
d_config_knobs["on_x8v"] = 200.000
d_config_knobs["on_sep8v"] = 0.000
d_config_knobs["phi_IR8"] = 180.000

# Octupoles
d_config_knobs["i_oct_b1"] = 300.0
d_config_knobs["i_oct_b2"] = 300.0

### leveling configuration

# Leveling in IP 1/5
d_config_leveling_ip1_5 = {"constraints": {}}
d_config_leveling_ip1_5["luminosity"] = 2.0e34
d_config_leveling_ip1_5["constraints"]["max_intensity"] = 1.8e11
d_config_leveling_ip1_5["constraints"]["max_PU"] = 70


# Define dictionary for the leveling settings
d_config_leveling = {
    "ip2": {},
    "ip8": {},
}

# Luminosity and particles


# Leveling parameters (ignored if skip_leveling is True)
d_config_leveling["ip2"]["separation_in_sigmas"] = 5
d_config_leveling["ip8"]["luminosity"] = 2.0e32

### Beam beam configuration

# Define dictionary for the beam beam settings
d_config_beambeam = {"mask_with_filling_pattern": {}}

# Beam settings
d_config_beambeam["num_particles_per_bunch"] = 1.15e11
d_config_beambeam["nemitt_x"] = 2.2e-6
d_config_beambeam["nemitt_y"] = 2.2e-6

# Filling scheme (in json format)
# The scheme should consist of a json file containing two lists of booleans (one for each beam),
# representing each bucket of the LHC.
filling_scheme_path = os.path.abspath(
    "master_jobs/filling_scheme/25ns_2464b_2452_1842_1821_236bpi_12inj_hybrid.json"
)

# Alternatively, one can get a fill directly from LPC from, e.g.:
# https://lpc.web.cern.ch/cgi-bin/fillTable.py?year=2023
# In this page, get the fill number of your fill of interest, and use it to replace the XXXX in the
# URL below before downloading:
# https://lpc.web.cern.ch/cgi-bin/schemeInfo.py?fill=XXXX&fmt=json
# Unfortunately, the format is not the same as the one used by defaults in xmask, but it should
# still be converted in the lines below (see with matteo.rufolo@cern.ch for questions, or if it
# doesn't work).
# %%
# Load filling scheme
if filling_scheme_path.endswith(".json"):
    with open(filling_scheme_path, "r") as fid:
        d_filling_scheme = json.load(fid)

# If the filling scheme is already in the correct format, do nothing
if "beam1" in d_filling_scheme.keys() and "beam2" in d_filling_scheme.keys():
    pass
# Otherwise, we need to reformat the file
else:
    # One can potentially use b1_array, b2_array to scan the bunches later
    b1_array, b2_array = reformat_filling_scheme_from_lpc_alt(filling_scheme_path)
    filling_scheme_path = filling_scheme_path.replace(".json", "_converted.json")


# Add to config file
d_config_beambeam["mask_with_filling_pattern"][
    "pattern_fname"
] = filling_scheme_path  # If None, a full fill is assumed

# Initialize bunch number to None (will be set later)
d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b1"] = None
d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b2"] = None

# Set this variable to False if you intend to scan the bunch number (but ensure both bunches indices
# are defined later)
check_bunch_number = True
if check_bunch_number:
    # Bunch number is ignored if pattern_fname is None (in which case the simulation considers all
    # bunch elements). It must be specified otherwise)
    # If the bunch number is None and pattern_name is defined, the bunch with the largest number of
    # long-range interactions will be used
    if d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b1"] is None:
        # Case the bunch number has not been provided
        worst_bunch_b1 = get_worst_bunch(
            filling_scheme_path, numberOfLRToConsider=26, beam="beam_1"
        )
        while d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b1"] is None:
            bool_inp = input(
                "The bunch number for beam 1 has not been provided. Do you want to use the bunch"
                " with the largest number of long-range interactions? It is the bunch number "
                + str(worst_bunch_b1)
                + " (y/n): "
            )
            if bool_inp == "y":
                d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b1"] = worst_bunch_b1
            elif bool_inp == "n":
                d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b1"] = int(
                    input("Please enter the bunch number for beam 1: ")
                )

    if d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b2"] is None:
        worst_bunch_b2 = get_worst_bunch(
            filling_scheme_path, numberOfLRToConsider=26, beam="beam_2"
        )
        # For beam 2, just select the worst bunch by default, as the tracking of b2 is not available yet anyway
        print(
            "The bunch number for beam 2 has not been provided. By default, the worst bunch is"
            " taken. It is the bunch number " + str(worst_bunch_b2)
        )

        d_config_beambeam["mask_with_filling_pattern"]["i_bunch_b2"] = worst_bunch_b2


# ==================================================================================================
# --- Generate dictionnary to encapsulate all base collider parameters (generation 2)
# ==================================================================================================
d_config_collider = {}

# Add tunes and chromas
d_config_collider["config_knobs_and_tuning"] = d_config_tune_and_chroma

# Add knobs
d_config_collider["config_knobs_and_tuning"]["knob_settings"] = d_config_knobs

# Add luminosity configuration
d_config_collider["config_lumi_leveling_ip1_5"] = d_config_leveling_ip1_5
d_config_collider["config_lumi_leveling"] = d_config_leveling

# Add beam beam configuration
d_config_collider["config_beambeam"] = d_config_beambeam

# ==================================================================================================
# --- Tracking parameters (generation 2)
#
# Below, the user defines the parameters for the tracking.
# ==================================================================================================
d_config_simulation = {}

# Number of turns to track
d_config_simulation["n_turns"] = 200

# Initial off-momentum
d_config_simulation["delta_max"] = 27.0e-5

# Beam to track (lhcb1 or lhcb2)
d_config_simulation["beam"] = "lhcb1"

# ==================================================================================================
# --- Dump collider and collider configuration
#
# Below, the user chooses if the gen 2 collider must be dumped, along with the corresponding
# configuration.
# ==================================================================================================
dump_collider = False
dump_config_in_collider = False

# ==================================================================================================
# --- Machine parameters being scanned (generation 2)
#
# Below, the user defines the grid for the machine parameters that must be scanned to find the
# optimal DA (e.g. tune, chroma, etc).
# ==================================================================================================
# Scan tune with step of 0.001 (need to round to correct for numpy numerical instabilities)
array_qx = np.round(np.arange(62.305, 62.330, 0.001), decimals=4)[:1]
array_qy = np.round(np.arange(60.305, 60.330, 0.001), decimals=4)[:1]

# In case one is doing a tune-tune scan, to decrease the size of the scan, we can ignore the
# working points too close to resonance. Otherwise just delete this variable in the loop at the end
# of the script
keep = 'all'  # 'lower_triangle', , "upper_triangle"
# ==================================================================================================
# --- Make tree for the simulations (generation 1)
#
# The tree is built as a hierarchy of dictionnaries. We add a first generation (named as the
# study being done) to the root. This first generation is used set the initial particle
# distribution, and build a collider with only the optics set.
# ==================================================================================================

# Build empty tree: first generation (later added to the root), and second generation
children = {"base_collider": {"config_particles": {}, "config_mad": {}, "children": {}}}

# Add particles distribution parameters to the first generation
children["base_collider"]["config_particles"] = d_config_particles

# Add base machine parameters to the first generation
children["base_collider"]["config_mad"] = d_config_mad


# ==================================================================================================
# --- Complete tree for the simulations (generation 2)
#
# We now set a second generation for the tree. This second generation contains the tracking
# parameters, as well as a default set of parameters for the colliders (defined above), that we
# mutate according to the parameters we want to scan.
# ! Caution when mutating the dictionnary in this function, you have to pass a deepcopy to children,
# ! otherwise the dictionnary will be mutated for all the children.
# ==================================================================================================
track_array = np.arange(d_config_particles["n_split"])
for idx_job, (track, qx, qy) in enumerate(itertools.product(track_array, array_qx, array_qy)):
    # If requested, ignore conditions below the upper diagonal as they can't be reached in the LHC
    if keep == "upper_triangle":
        if qy < (qx - 2 + 0.0039):  # 0.039 instead of 0.04 to avoid rounding errors
            continue
    elif keep == "lower_triangle":
        if qy >= (qx - 2 - 0.0039):
            continue
    else:
        pass

    # Mutate the appropriate collider parameters
    for beam in ["lhcb1", "lhcb2"]:
        d_config_collider["config_knobs_and_tuning"]["qx"][beam] = float(qx)
        d_config_collider["config_knobs_and_tuning"]["qy"][beam] = float(qy)

    # Complete the dictionnary for the tracking
    d_config_simulation["particle_file"] = f"../particles_new/{track:02}.parquet"
    d_config_simulation["collider_file"] = f"../collider/collider.json"

    # Add a child to the second generation, with all the parameters for the collider and tracking
    children["base_collider"]["children"][f"xtrack_{idx_job:04}"] = {
        "config_simulation": copy.deepcopy(d_config_simulation),
        "config_collider": copy.deepcopy(d_config_collider),
        "log_file": "tree_maker.log",
        "dump_collider": dump_collider,
        "dump_config_in_collider": dump_config_in_collider,
    }

# ==================================================================================================
# --- Simulation configuration
# ==================================================================================================
# Load the tree_maker simulation configuration
config = yaml.safe_load(open("config.yaml"))

# # Set the root children to the ones defined above
config["root"]["children"] = children

# Set miniconda environment path in the config
config["root"]["setup_env_script"] = "/afs/cern.ch/work/a/aradosla/private/miniforge3/bin/activate"


# Recursively define the context for the simulations
def set_context(children, idx_gen, config):
    for child in children.values():
        child["context"] = config["root"]["generations"][idx_gen]["context"]
        if "children" in child.keys():
            set_context(child["children"], idx_gen + 1, config)


set_context(children, 1, config)
# %%
# ==================================================================================================
# --- Build tree and write it to the filesystem
# ==================================================================================================
# Define study name
study_name = "example_tunescan"

# Creade folder that will contain the tree
if not os.path.exists("scans/" + study_name):
    os.makedirs("scans/" + study_name)

# Move to the folder that will contain the tree
os.chdir("scans/" + study_name)

# Clean the id_job file
id_job_file_path = "id_job.yaml"
if os.path.isfile(id_job_file_path):
    os.remove(id_job_file_path)

# Create tree object
start_time = time.time()
root = initialize(config)
print("Done with the tree creation.")
print("--- %s seconds ---" % (time.time() - start_time))

# Check if htcondor is the configuration
if "htc" in config["root"]["generations"][2]["run_on"]:
    generate_run = generate_run_sh_htc
else:
    generate_run = generate_run_sh

# From python objects we move the nodes to the filesystem.
start_time = time.time()
root.make_folders(generate_run)
print("The tree folders are ready.")
print("--- %s seconds ---" % (time.time() - start_time))

# %%
