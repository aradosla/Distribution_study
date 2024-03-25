"""This script is used to configure the collider and track the particles. Functions in this script
are called sequentially, in the order in which they are defined. Modularity has been favored over 
simple scripting for reproducibility, to allow rebuilding the collider from a different program 
(e.g. dahsboard)."""

# %%
# ==================================================================================================
# --- Imports
# ==================================================================================================
# Import standard library modules
import json
import logging
import os
import time
from pathlib import Path

# Import third-party modules
import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker

# Import user-defined modules
import xmask as xm
import xobjects as xo
import xtrack as xt
from misc import (
    compute_PU,
    generate_orbit_correction_setup,
    luminosity_leveling,
    luminosity_leveling_ip1_5,
)

# Initialize yaml reader
ryaml = ruamel.yaml.YAML()


# ==================================================================================================
# --- Function for tree_maker tagging
# ==================================================================================================
def tree_maker_tagging(config, tag="started"):
    # Start tree_maker logging if log_file is present in config
    if tree_maker is not None and "log_file" in config:
        tree_maker.tag_json.tag_it(config["log_file"], tag)
    else:
        logging.warning("tree_maker loging not available")


# ==================================================================================================
# --- Function to get context
# ==================================================================================================
def get_context(configuration):
    if configuration["context"] == "cupy":
        context = xo.ContextCupy()
    elif configuration["context"] == "opencl":
        context = xo.ContextPyopencl()
    elif configuration["context"] == "cpu":
        context = xo.ContextCpu()
    else:
        logging.warning("context not recognized, using cpu")
        context = xo.ContextCpu()
    return context


# ==================================================================================================
# --- Functions to read configuration files and generate configuration files for orbit correction
# ==================================================================================================
def read_configuration(config_path="config.yaml"):
    # Read configuration for simulations
    with open(config_path, "r") as fid:
        config = ryaml.load(fid)

    # Also read configuration from previous generation
    try:
        with open("../" + config_path, "r") as fid:
            config_gen_1 = ryaml.load(fid)
    except:
        with open("../1_build_distr_and_collider/" + config_path, "r") as fid:
            config_gen_1 = ryaml.load(fid)

    config_mad = config_gen_1["config_mad"]
    return config, config_mad


def generate_configuration_correction_files(output_folder="correction"):
    # Generate configuration files for orbit correction
    correction_setup = generate_orbit_correction_setup()
    os.makedirs(output_folder, exist_ok=True)
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"{output_folder}/corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)


# ==================================================================================================
# --- Function to install beam-beam
# ==================================================================================================
def install_beam_beam(collider, config_collider):
    # Load config
    config_bb = config_collider["config_beambeam"]

    # Install beam-beam lenses (inactive and not configured)
    collider.install_beambeam_interactions(
        clockwise_line="lhcb1",
        anticlockwise_line="lhcb2",
        ip_names=["ip1", "ip2", "ip5", "ip8"],
        delay_at_ips_slots=[0, 891, 0, 2670],
        num_long_range_encounters_per_side=config_bb["num_long_range_encounters_per_side"],
        num_slices_head_on=config_bb["num_slices_head_on"],
        harmonic_number=35640,
        bunch_spacing_buckets=config_bb["bunch_spacing_buckets"],
        sigmaz=config_bb["sigma_z"],
    )

    return collider, config_bb


# ==================================================================================================
# --- Function to match knobs and tuning
# ==================================================================================================
def set_knobs(config_collider, collider):
    # Read knobs and tuning settings from config file
    conf_knobs_and_tuning = config_collider["config_knobs_and_tuning"]

    # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
    # experimental magnets, etc.)
    for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
        collider.vars[kk] = vv

    return collider, conf_knobs_and_tuning


def match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True):
    # Tunings
    for line_name in ["lhcb1", "lhcb2"]:
        knob_names = conf_knobs_and_tuning["knob_names"][line_name]

        targets = {
            "qx": conf_knobs_and_tuning["qx"][line_name],
            "qy": conf_knobs_and_tuning["qy"][line_name],
            "dqx": conf_knobs_and_tuning["dqx"][line_name],
            "dqy": conf_knobs_and_tuning["dqy"][line_name],
        }

        xm.machine_tuning(
            line=collider[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=match_linear_coupling_to_zero,
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
            line_co_ref=collider[line_name + "_co_ref"],
            co_corr_config=conf_knobs_and_tuning["closed_orbit_correction"][line_name],
        )

    return collider


# ==================================================================================================
# --- Function to compute the number of collisions in the IPs (used for luminosity leveling)
# ==================================================================================================
def compute_collision_from_scheme(config_bb):
    # Get the filling scheme path (in json or csv format)
    filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

    # Load the filling scheme
    if filling_scheme_path.endswith(".json"):
        with open(filling_scheme_path, "r") as fid:
            filling_scheme = json.load(fid)
    else:
        raise ValueError(
            f"Unknown filling scheme file format: {filling_scheme_path}. It you provided a csv"
            " file, it should have been automatically convert when running the script"
            " 001_make_folders.py. Something went wrong."
        )

    # Extract booleans beam arrays
    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    # Assert that the arrays have the required length, and do the convolution
    assert len(array_b1) == len(array_b2) == 3564
    n_collisions_ip1_and_5 = array_b1 @ array_b2
    n_collisions_ip2 = np.roll(array_b1, 891) @ array_b2
    n_collisions_ip8 = np.roll(array_b1, 2670) @ array_b2

    return n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8


# ==================================================================================================
# --- Function to do the Levelling
# ==================================================================================================
def do_levelling(
    config_collider,
    config_bb,
    n_collisions_ip2,
    n_collisions_ip8,
    collider,
    n_collisions_ip1_and_5,
    crab,
):
    # Read knobs and tuning settings from config file (already updated with the number of collisions)
    config_lumi_leveling = config_collider["config_lumi_leveling"]

    # Update the number of bunches in the configuration file
    config_lumi_leveling["ip2"]["num_colliding_bunches"] = int(n_collisions_ip2)
    config_lumi_leveling["ip8"]["num_colliding_bunches"] = int(n_collisions_ip8)

    # Initial intensity
    initial_I = config_bb["num_particles_per_bunch"]

    # First level luminosity in IP 1/5 changing the intensity
    if "config_lumi_leveling_ip1_5" in config_collider:
        if not config_collider["config_lumi_leveling_ip1_5"]["skip_leveling"]:
            print("Leveling luminosity in IP 1/5 varying the intensity")
            # Update the number of bunches in the configuration file
            config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"] = int(
                n_collisions_ip1_and_5
            )

            # Do the levelling
            try:
                I = luminosity_leveling_ip1_5(
                    collider,
                    config_collider,
                    config_bb,
                    crab=crab,
                )
            except ValueError:
                print("There was a problem during the luminosity leveling in IP1/5... Ignoring it.")
                I = config_bb["num_particles_per_bunch"]

            config_bb["num_particles_per_bunch"] = float(I)

    # Set up the constraints for lumi optimization in IP8
    additional_targets_lumi = []
    if "constraints" in config_lumi_leveling["ip8"]:
        for constraint in config_lumi_leveling["ip8"]["constraints"]:
            obs, beam, sign, val, at = constraint.split("_")
            if sign == "<":
                ineq = xt.LessThan(float(val))
            elif sign == ">":
                ineq = xt.GreaterThan(float(val))
            else:
                raise ValueError(f"Unsupported sign for luminosity optimization constraint: {sign}")
            target = xt.Target(obs, ineq, at=at, line=beam, tol=1e-6)
            additional_targets_lumi.append(target)

    # Then level luminosity in IP 2/8 changing the separation
    collider = luminosity_leveling(
        collider,
        config_lumi_leveling=config_lumi_leveling,
        config_beambeam=config_bb,
        additional_targets_lumi=additional_targets_lumi,
        crab=crab,
    )

    # Update configuration
    config_bb["num_particles_per_bunch_before_optimization"] = float(initial_I)
    config_collider["config_lumi_leveling"]["ip2"]["final_on_sep2h"] = float(
        collider.vars["on_sep2h"]._value
    )
    config_collider["config_lumi_leveling"]["ip2"]["final_on_sep2v"] = float(
        collider.vars["on_sep2v"]._value
    )
    config_collider["config_lumi_leveling"]["ip8"]["final_on_sep8h"] = float(
        collider.vars["on_sep8h"]._value
    )
    config_collider["config_lumi_leveling"]["ip8"]["final_on_sep8v"] = float(
        collider.vars["on_sep8v"]._value
    )

    return collider, config_collider


# ==================================================================================================
# --- Function to add linear coupling
# ==================================================================================================
def add_linear_coupling(conf_knobs_and_tuning, collider, config_mad):
    # Get the version of the optics
    version_hllhc = config_mad["ver_hllhc_optics"]
    version_run = config_mad["ver_lhc_run"]

    # Add linear coupling as the target in the tuning of the base collider was 0
    # (not possible to set it the target to 0.001 for now)
    if version_run == 3.0:
        collider.vars["cmrs.b1_sq"] += conf_knobs_and_tuning["delta_cmr"]
        collider.vars["cmrs.b2_sq"] += conf_knobs_and_tuning["delta_cmr"]
    elif version_hllhc == 1.6 or version_hllhc == 1.5:
        collider.vars["c_minus_re_b1"] += conf_knobs_and_tuning["delta_cmr"]
        collider.vars["c_minus_re_b2"] += conf_knobs_and_tuning["delta_cmr"]
    else:
        raise ValueError(f"Unknown version of the optics/run: {version_hllhc}, {version_run}.")

    return collider


# ==================================================================================================
# --- Function to assert that tune, chromaticity and linear coupling are correct before beam-beam
#     configuration
# ==================================================================================================
def assert_tune_chroma_coupling(collider, conf_knobs_and_tuning):
    for line_name in ["lhcb1", "lhcb2"]:
        tw = collider[line_name].twiss()
        assert np.isclose(tw.qx, conf_knobs_and_tuning["qx"][line_name], atol=1e-4), (
            f"tune_x is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['qx'][line_name]}, got {tw.qx}"
        )
        assert np.isclose(tw.qy, conf_knobs_and_tuning["qy"][line_name], atol=1e-4), (
            f"tune_y is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['qy'][line_name]}, got {tw.qy}"
        )
        assert np.isclose(
            tw.dqx,
            conf_knobs_and_tuning["dqx"][line_name],
            rtol=1e-2,
        ), (
            f"chromaticity_x is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['dqx'][line_name]}, got {tw.dqx}"
        )
        assert np.isclose(
            tw.dqy,
            conf_knobs_and_tuning["dqy"][line_name],
            rtol=1e-2,
        ), (
            f"chromaticity_y is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['dqy'][line_name]}, got {tw.dqy}"
        )

        assert np.isclose(
            tw.c_minus,
            conf_knobs_and_tuning["delta_cmr"],
            atol=5e-3,
        ), (
            f"linear coupling is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['delta_cmr']}, got {tw.c_minus}"
        )


# ==================================================================================================
# --- Function to configure beam-beam
# ==================================================================================================
def configure_beam_beam(collider, config_bb):
    collider.configure_beambeam_interactions(
        num_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    # Configure filling scheme mask and bunch numbers
    if "mask_with_filling_pattern" in config_bb:
        # Initialize filling pattern with empty values
        filling_pattern_cw = None
        filling_pattern_acw = None

        # Initialize bunch numbers with empty values
        i_bunch_cw = None
        i_bunch_acw = None

        if "pattern_fname" in config_bb["mask_with_filling_pattern"]:
            # Fill values if possible
            if config_bb["mask_with_filling_pattern"]["pattern_fname"] is not None:
                fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
                with open(fname, "r") as fid:
                    filling = json.load(fid)
                filling_pattern_cw = filling["beam1"]
                filling_pattern_acw = filling["beam2"]

                # Only track bunch number if a filling pattern has been provided
                if "i_bunch_b1" in config_bb["mask_with_filling_pattern"]:
                    i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
                if "i_bunch_b2" in config_bb["mask_with_filling_pattern"]:
                    i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]

                # Note that a bunch number must be provided if a filling pattern is provided
                # Apply filling pattern
                collider.apply_filling_pattern(
                    filling_pattern_cw=filling_pattern_cw,
                    filling_pattern_acw=filling_pattern_acw,
                    i_bunch_cw=i_bunch_cw,
                    i_bunch_acw=i_bunch_acw,
                )
    return collider


# ==================================================================================================
# --- Function to compute luminosity once the collider is configured
# ==================================================================================================
def record_final_luminosity(collider, config_bb, l_n_collisions, crab):
    # Get the final luminoisty in all IPs
    twiss_b1 = collider["lhcb1"].twiss()
    twiss_b2 = collider["lhcb2"].twiss()
    l_lumi = []
    l_PU = []
    l_ip = ["ip1", "ip2", "ip5", "ip8"]
    for n_col, ip in zip(l_n_collisions, l_ip):
        try:
            L = xt.lumi.luminosity_from_twiss(
                n_colliding_bunches=n_col,
                num_particles_per_bunch=config_bb["num_particles_per_bunch"],
                ip_name=ip,
                nemitt_x=config_bb["nemitt_x"],
                nemitt_y=config_bb["nemitt_y"],
                sigma_z=config_bb["sigma_z"],
                twiss_b1=twiss_b1,
                twiss_b2=twiss_b2,
                crab=crab,
            )
            PU = compute_PU(L, n_col, twiss_b1["T_rev0"])
        except:
            print(f"There was a problem during the luminosity computation in {ip}... Ignoring it.")
            L = 0
            PU = 0
        l_lumi.append(L)
        l_PU.append(PU)

    # Update configuration
    for ip, L, PU in zip(l_ip, l_lumi, l_PU):
        config_bb[f"luminosity_{ip}_after_optimization"] = float(L)
        config_bb[f"Pile-up_{ip}_after_optimization"] = float(PU)

    return config_bb


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def configure_collider(
    config,
    config_mad,
    context,
    save_collider=True,
    #save_collider=False,
    save_config=False,
    return_collider_before_bb=False,
    config_path="config.yaml",
):
    # Generate configuration files for orbit correction
    generate_configuration_correction_files()

    # Get configurations
    config_sim = config["config_simulation"]
    config_collider = config["config_collider"]

    # Rebuild collider
    collider = xt.Multiline.from_json(config_sim["collider_file"])

    # Install beam-beam
    collider, config_bb = install_beam_beam(collider, config_collider)

    # Build trackers
    # For now, start with CPU tracker due to a bug with Xsuite
    # Refer to issue https://github.com/xsuite/xsuite/issues/450
    collider.build_trackers()  # (_context=context)

    # Set knobs
    collider, conf_knobs_and_tuning = set_knobs(config_collider, collider)

    # Match tune and chromaticity
    collider = match_tune_and_chroma(
        collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True
    )
 
    # Compute the number of collisions in the different IPs
    (
        n_collisions_ip1_and_5,
        n_collisions_ip2,
        n_collisions_ip8,
    ) = compute_collision_from_scheme(config_bb)

    # Get crab cavities
    crab = False
    if "on_crab1" in config_collider["config_knobs_and_tuning"]["knob_settings"]:
        crab_val = float(config_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab1"])
        if abs(crab_val) > 0:
            crab = True

    # Do the leveling if requested
    if "config_lumi_leveling" in config_collider and not config_collider["skip_leveling"]:
        collider, config_collider = do_levelling(
            config_collider,
            config_bb,
            n_collisions_ip2,
            n_collisions_ip8,
            collider,
            n_collisions_ip1_and_5,
            crab,
        )

    else:
        print(
            "No leveling is done as no configuration has been provided, or skip_leveling"
            " is set to True."
        )

    # Add linear coupling
    collider = add_linear_coupling(conf_knobs_and_tuning, collider, config_mad)

    # Rematch tune and chromaticity
    collider = match_tune_and_chroma(
        collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=False
    )

    # Assert that tune, chromaticity and linear coupling are correct one last time
    assert_tune_chroma_coupling(collider, conf_knobs_and_tuning)

    # Return twiss and survey before beam-beam if requested
    if return_collider_before_bb:
        print("Saving collider before beam-beam configuration")
        collider_before_bb = xt.Multiline.from_dict(collider.to_dict())

    if not config_bb["skip_beambeam"]:
        # Configure beam-beam
        collider = configure_beam_beam(collider, config_bb)

    # Update configuration with luminosity now that bb is known
    l_n_collisions = [
        n_collisions_ip1_and_5,
        n_collisions_ip2,
        n_collisions_ip1_and_5,
        n_collisions_ip8,
    ]
    config_bb = record_final_luminosity(collider, config_bb, l_n_collisions, crab)

    # Drop update configuration
    with open(config_path, "w") as fid:
        ryaml.dump(config, fid)

    if save_collider:
        # Save the final collider before tracking
        print('Saving "collider.json')
        if save_config:
            config_dict = {
                "config_mad": config_mad,
                "config_collider": config_collider,
            }
            collider.metadata = config_dict
        # Dump collider
        collider.to_json("collider.json")

    if return_collider_before_bb:
        return collider, config_sim, config_bb, collider_before_bb
    else:
        return collider, config_sim, config_bb



# ==================================================================================================
# Function for reading of the distribution
# ==================================================================================================

def prepare_particle_distribution(collider, context, config_sim):
    beam = config_sim["beam"]

    #particle_df = pd.read_parquet(config_sim["particle_file"])
    particle_df = pd.read_parquet(config_sim["particle_file"])

    print(particle_df.x)


    particles = collider[beam].build_particles(
        x=particle_df.x.values,
        y=particle_df.y.values,
        px = particle_df.px.values,
        py = particle_df.py.values,
        zeta = particle_df.zeta.values,
        delta=particle_df.delta.values,
        _context=context,
    )


    particle_id = particle_df.particle_id.values

    return particles, particle_id

#particles, particle_id = prepare_particle_distribution(collider,ctx, 'particles_new/00.parquet', 'lhcb1')



# ==================================================================================================
# --- Function to do the tracking
# ==================================================================================================
def track(collider, particles, config_sim, config_bb, save_input_particles=False):
    # Get beam being tracked
    config, config_mad = read_configuration("config.yaml")
    context = get_context(config)
    beam = config_sim["beam"]

    # Optimize line for tracking (not working for now)
    # collider[beam].optimize_for_tracking()

    # Save initial coordinates if requested
    if save_input_particles:
        pd.DataFrame(particles.to_dict()).to_parquet("input_particles_new.parquet")              # here save the initial distribution

    # Track
    num_turns = config_sim["n_turns"]
    a = time.time()

    
    collider.discard_trackers()
    line = collider[beam]

    
    '''
    f = 50
    phi = 0
    A = 0
    sampling_frequency = 11245.5
    
    total_time = num_turns / sampling_frequency
    time_sim = np.arange(0, total_time, 1/sampling_frequency)
    samples = A * np.sin(2*np.pi*f*time_sim + phi)

    exciter = xt.Exciter(_context = context,
        samples = samples,
        sampling_frequency = sampling_frequency,
        #duration = 0.05,  # defaults to waveform duration   in sec, checked with the number of turns 0.05*11245.5=562.275
        duration= num_turns/sampling_frequency,
        frev = sampling_frequency,
        #start_turn = num_turns/10,  # default, seconds
        #start_turn = 500*1/sampling_frequency ,  
        #knl = [0.00000001],  # default, no kick
        knl = [0.00001]
        # default, no kick
        #knl = [1],
        #ksl = []
    )
  
    line.insert_element(
        element = exciter,
        name = 'RF_KO_EXCITER',
        index ='mb.b9r3.b1',
    )
    '''
    line.build_tracker(_context = context)
    #collider.build_trackers(_context=context)

    
    # ================================= New part ==================================
    x_tot =           []
    y_tot =           []
    px_tot =          []
    py_tot =          []
    zeta_tot =        []
    turns_tot =       []
    particle_id_tot = []
    pzeta_tot =       []


    x_phys =          []
    y_phys =          []
    zeta_phys =       []
    px_phys =         []
    py_phys =         []
    pzeta_phys =      []
    state_all =       []

    for i in range(num_turns):
        line.track(particles, num_turns = 1, turn_by_turn_monitor=True, freeze_longitudinal=True)
        x_phys.append(np.copy(particles.x))
        y_phys.append(np.copy(particles.y))
        zeta_phys.append(np.copy(particles.zeta))
        px_phys.append(np.copy(particles.px))
        py_phys.append(np.copy(particles.py))
        pzeta_phys.append(np.copy(particles.delta))
        state_all.append(np.copy(particles.state))
        coord = line.twiss().get_normalized_coordinates(particles, nemitt_x = config_bb["nemitt_x"], #m*rad
        nemitt_y = config_bb['nemitt_y']) #m*rad)
        x_tot.append(coord.x_norm)
        y_tot.append(coord.y_norm)
        zeta_tot.append(coord.zeta_norm)
        px_tot.append(coord.px_norm)
        py_tot.append(coord.py_norm)
        pzeta_tot.append(coord.pzeta_norm)
        turns_tot.append(np.ones(len(coord.x_norm))*i)
        particle_id_tot.append(coord.particle_id)
    
    x_physflat = [item for sublist in x_phys for item in sublist]
    y_physflat = [item for sublist in y_phys for item in sublist]
    zeta_physflat = [item for sublist in zeta_phys for item in sublist]
    px_physflat = [item for sublist in px_phys for item in sublist]
    py_physflat = [item for sublist in py_phys for item in sublist]
    pzeta_physflat = [item for sublist in pzeta_phys for item in sublist]
    state_allflat = [item for sublist in state_all for item in sublist]

    x_normflat = [item for sublist in x_tot for item in sublist]
    y_normflat = [item for sublist in y_tot for item in sublist]
    zeta_normflat = [item for sublist in zeta_tot for item in sublist]
    px_normflat = [item for sublist in px_tot for item in sublist]
    py_normflat = [item for sublist in py_tot for item in sublist]
    pzeta_normflat = [item for sublist in pzeta_tot for item in sublist]

    particle_id_flat = [item for sublist in particle_id_tot for item in sublist]
    turns_flat = [item for sublist in turns_tot for item in sublist]
    dictionary = {"particle_id": particle_id_flat, "at_turn": turns_flat, "state":state_allflat, "x_phys": x_physflat, "y_phys": y_physflat, "zeta_phys": zeta_physflat, "px_phys": px_physflat, "py_phys": py_physflat, "pzeta_phys": pzeta_physflat, "x_norm": x_normflat, "y_norm": y_normflat, "zeta_norm": zeta_normflat, "px_norm": px_normflat, "py_norm": py_normflat, "pzeta_norm": pzeta_normflat}
    result_phys = pd.DataFrame(dictionary)

    gamma_rel = particles.gamma0[0]
    betx_rel = particles.beta0[0]
    tw0 = line.twiss()

    # Emittance

    geomx_all_std = []
    geomy_all_std = []
    normx_all_std = []
    normy_all_std = []


    for turn in range(num_turns):
        #print(turn)
        sigma_delta = float(np.std(result_phys[result_phys['at_turn'] == turn].pzeta_phys))
        sigma_x = float(np.std(result_phys[result_phys['at_turn'] == turn].x_phys))
        sigma_y = float(np.std(result_phys[result_phys['at_turn'] == turn].y_phys))
        
        geomx_emittance = (sigma_x**2-(tw0[:,0]["dx"][0]*sigma_delta)**2)/tw0[:,0]["betx"][0]
        
        normx_emittance = geomx_emittance*(gamma_rel*betx_rel)
        geomx_all_std.append(geomx_emittance)
        normx_all_std.append(normx_emittance)

        geomy_emittance = (sigma_y**2-(tw0[:,0]["dy"][0]*sigma_delta)**2)/tw0[:,0]["bety"][0]
        normy_emittance = geomy_emittance*(gamma_rel*betx_rel)
        geomy_all_std.append(geomy_emittance)
        normy_all_std.append(normy_emittance)

    pd.set_option('float_format', '{:.10f}'.format) 
    for i in range(num_turns):
        mask = result_phys["at_turn"] == i
        result_phys.loc[mask, 'norm_emitx'] = normx_all_std[i] * np.ones(mask.sum())
        result_phys.loc[mask, 'norm_emity'] = normy_all_std[i] * np.ones(mask.sum())


    #line.track(particles, turn_by_turn_monitor=True, num_turns=num_turns)         # tracking of the distribution 
    #particles = collider[beam].record_last_track
    b = time.time()

    
    print(f"Elapsed time: {b-a} s")
    #print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

    return result_phys


# ==================================================================================================
# --- Main function for collider configuration and tracking
# ==================================================================================================
def configure_and_track(config_path="config.yaml"):
    # Get configuration
    config, config_mad = read_configuration(config_path)
    path = config['log_file']
    folder, tail = os.path.split(path)
    new_folder = 'Noise_sim_try'
    parent_folder, current_folder = os.path.split(folder)
    new_directory = f"/eos/user/a/aradosla/SWAN_projects/{new_folder}/{current_folder}"
    Path(new_directory).mkdir(parents=True, exist_ok=True)

    # Get context
    context = get_context(config)

    # Tag start of the job
    tree_maker_tagging(config, tag="started")

    # Configure collider (not saved, since it may trigger overload of afs)
    collider, config_sim, config_bb = configure_collider(
        config,
        config_mad,
        context,
        save_collider=config["dump_collider"],
        save_config=config["dump_config_in_collider"],
        config_path=config_path,
    )

    # Reset the tracker to go to GPU if needed
    if config["context"] in ["cupy", "opencl"]:
        collider.discard_trackers()
        collider.build_trackers(_context=context)

    # Prepare particle distribution
    print('Now preparing distribution!')

    particles, particle_id = prepare_particle_distribution(collider, context, config_sim)
    
    # Track
    print('Now tracking!')
    particles = track(collider, particles, config_sim, config_bb)

    # Get particles dictionnary
    #particles_dict = particles.to_dict()
    #particles_dict["particle_id"] = particle_id

    
    # Save output


    #pd.DataFrame(particles_dict['data']).to_parquet(f"/eos/user/a/aradosla/SWAN_projects/Noise_sim/{current_folder}/output_particles_new.parquet")
    particles.to_parquet(f"/eos/user/a/aradosla/SWAN_projects/{new_folder}/{current_folder}/output_particles_new.parquet")
    print('The parquet should be saved')
    # Remote the correction folder, and potential C files remaining
    try:
        os.system("rm -rf correction")
        os.system("rm -f *.cc")
    except:
        pass

    # Tag end of the job
    tree_maker_tagging(config, tag="completed")


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    configure_and_track()

# %%
