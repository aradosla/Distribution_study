"""This script is used to build the base collider with Xmask, configuring only the optics. Functions
in this script are called sequentially."""
# %%
# ==================================================================================================
# --- Imports
# ==================================================================================================
# %%
# Import standard library modules
import itertools
import json
import logging
import os
import shutil

# Import third-party modules
import numpy as np

# Import user-defined modules
import optics_specific_tools as ost
import pandas as pd
import tree_maker
import xmask as xm
import xpart as xp
import xmask.lhc as xlhc
import xobjects as xo
import yaml
from cpymad.madx import Madx

#%%
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
# --- Function to load configuration file
# ==================================================================================================
def load_configuration(config_path="config.yaml"):
    # Load configuration
    with open(config_path, "r") as fid:
        configuration = yaml.safe_load(fid)

    # Get configuration for the particles distribution and the collider separately
    config_particles = configuration["config_particles"]
    config_mad = configuration["config_mad"]

    return configuration, config_particles, config_mad


# START NEW
'''
def parameters(n_particles, sigma_d):
    
    n_part = n_particles
    
    x  = np.zeros(n_part)
    px = np.zeros(n_part)
    y  = np.zeros(n_part)
    py = np.zeros(n_part)
    z = np.zeros(n_part)

    dp = np.random.uniform(0.1*sigma_d,3.1*sigma_d,n_part)
    return  x, px, y, py, z, dp


def cmp_weights(df):
    
    r2 = df['x']**2 + df['px']**2 + df['y']**2 + df['py']**2
    w = np.exp(-r2/2.)
    #r2_l = df['z']**2 + df['dp']**2
    #w *=np.exp(r2_l/2.)
    w/=np.sum(w)
    return w


def generate_pseudoKV_xpyp(i, config_particles):
  not_generated = True
  n_part = config_particles['N_particles']
  sigma_z = config_particles['sigma_z']
  n_sigma = config_particles['n_sigma']
  x, px, y, py, z, dp = parameters(n_part, sigma_z)
  while not_generated:
    u = np.random.normal(size=4)
    r = np.sqrt(np.sum(u**2))
    u *= n_sigma/r
    v = np.random.normal(size=4)
    r = np.sqrt(np.sum(v**2))
    v *= n_sigma/r
    R2 = u[0]**2 + u[1]**2 + v[0]**2 + v[1]**2
    if R2 <= n_sigma**2:
        x[i]  = u[0]
        px[i] = u[1]
        y[i]  = v[0]
        py[i] = v[1]
        not_generated = False
  return x, y, px, py, z, dp


def df_colored_func(n_part):
    n_sigma, x, px, y, py, z, dp = parameters(n_part)
    list(map(generate_pseudoKV_xpyp, range(n_part)))
    df = pd.DataFrame({'x': x , 'y': y, 'px': px, 'py': py, 'z': z, 'dp': dp})
    df['weights'] = cmp_weights(df)
    return df

# =================================================================================================================================
# Colored gaussian
def generate_matched_gaussian_bunch_colored(num_particles,
                                    nemitt_x, nemitt_y, sigma_z,
                                    total_intensity_particles=None,
                                    particle_on_co=None,
                                    R_matrix=None,
                                    circumference=None,
                                    momentum_compaction_factor=None,
                                    rf_harmonic=None,
                                    rf_voltage=None,
                                    rf_phase=None,
                                    p_increment=0.,
                                    tracker=None,
                                    line=None,
                                    particle_ref=None,
                                    particles_class=None,
                                    engine=None,
                                    _context=None, _buffer=None, _offset=None,
                                    **kwargs, # They are passed to build_particles
                                    ):

    """
    Generate a matched Gaussian bunch.

    Parameters
    ----------
    line : xpart.Line
        Line for which the bunch is generated.
    num_particles : int
        Number of particles to be generated.
    nemitt_x : float
        Normalized emittance in the horizontal plane (in m rad).
    nemitt_y : float
        Normalized emittance in the vertical plane (in m rad).
    sigma_z : float
        RMS bunch length in meters.
    total_intensity_particles : float
        Total intensity of the bunch in particles.

    Returns
    -------
    part : xpart.Particles
        Particles object containing the generated particles.

    """

    if line is not None and tracker is not None:
        raise ValueError(
            'line and tracker cannot be provided at the same time.')

    if tracker is not None:
        print(
            "The argument tracker is deprecated. Please use line instead.",
            DeprecationWarning)
        line = tracker.line

    if line is not None:
        assert line.tracker is not None, ("The line has no tracker. Please use "
                                          "`Line.build_tracker()`")

    if (particle_ref is not None and particle_on_co is not None):
        raise ValueError("`particle_ref` and `particle_on_co`"
                " cannot be provided at the same time")

    if particle_ref is None:
        if particle_on_co is not None:
            particle_ref = particle_on_co
        elif line is not None and line.particle_ref is not None:
            particle_ref = line.particle_ref
        else:
            raise ValueError(
                "`line`, `particle_ref` or `particle_on_co` must be provided!")

    zeta, delta = xp.generate_longitudinal_coordinates(
            distribution='gaussian',
            num_particles=num_particles,
            particle_ref=(particle_ref if particle_ref is not None
                          else particle_on_co),
            line=line,
            circumference=circumference,
            momentum_compaction_factor=momentum_compaction_factor,
            rf_harmonic=rf_harmonic,
            rf_voltage=rf_voltage,
            rf_phase=rf_phase,
            p_increment=p_increment,
            sigma_z=sigma_z,
            engine=engine,
            **kwargs)

    assert len(zeta) == len(delta) == num_particles
    df_colored = df_colored_func(num_particles)
    x_norm = df_colored['x'].values
    px_norm = df_colored['px'].values
    y_norm = df_colored['y'].values
    py_norm = df_colored['py'].values

    if total_intensity_particles is None:
        # go to particles.weight = 1
        total_intensity_particles = num_particles


    part = xp.build_particles(_context=_context, _buffer=_buffer, _offset=_offset,
                      R_matrix=R_matrix,
                      particles_class=particles_class,
                      particle_on_co=particle_on_co,
                      particle_ref=(
                          particle_ref if particle_on_co is  None else None),
                      line=line,
                      zeta=zeta, delta=delta,
                      x_norm=x_norm, px_norm=px_norm,
                      y_norm=y_norm, py_norm=py_norm,
                      nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                      weight=total_intensity_particles/num_particles,
                      **kwargs)
    return part

def build_particle_distribution(config_mad, config_particles, collider):
    # Define radius distribution
    #r_min = config_particles["r_min"]
    #r_max = config_particles["r_max"]
    #n_r = config_particles["n_r"]
    #radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

    # Filter out particles with low and high amplitude to accelerate simulation
    # radial_list = radial_list[(radial_list >= 4.5) & (radial_list <= 7.5)]

    # Define angle distribution
    #n_angles = config_particles["n_angles"]
    #theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]
    N_particles = config_particles['N_particles'] #int(1e6)


    bunch_intensity = config_particles['bunch_intensity']
    normal_emitt_x = config_particles['norm_emitt_x'] #m*rad
    normal_emitt_y = config_particles['norm_emitt_y'] #m*rad
    sigma_z = config_particles['sigma_z']

    particle_ref = xp.Particles(
                        mass0=xp.PROTON_MASS_EV, q0=1, energy0=config_mad['beam_config']['lhcb1']['beam_energy_tot'])
    gaussian_bunch = generate_matched_gaussian_bunch_colored(
            num_particles = N_particles, total_intensity_particles = bunch_intensity,
            nemitt_x = normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z = sigma_z,
            particle_ref = particle_ref,
            line = collider['lhcb1'])
    # Define particle distribution as a cartesian product of the above
    particle_list = [
        (particle_id, x, y, px, py, zeta, delta)
        for particle_id, (x, y, px, py, zeta, delta) in enumerate(zip(gaussian_bunch.x, gaussian_bunch.y, gaussian_bunch.px, gaussian_bunch.py, gaussian_bunch.zeta, gaussian_bunch.delta))
    ]
    print('first',particle_list)
    # Split distribution into several chunks for parallelization
    n_split = config_particles['n_split']
    particle_list = np.array(np.array_split(particle_list, n_split))
    array_of_lists = np.array([arr.tolist() for arr in particle_list])
    particle_list = array_of_lists
    print('second',particle_list)

    # Return distribution
    return particle_list

''' 
# ============================================================================================================
# Gaussian!!! Rewrite to change the distribution as a variable

def build_particle_distribution(config_mad, config_particles, collider):
    # Define radius distribution
    #r_min = config_particles["r_min"]
    #r_max = config_particles["r_max"]
    #n_r = config_particles["n_r"]
    #radial_list = np.linspace(r_min, r_max, n_r, endpoint=False)

    # Filter out particles with low and high amplitude to accelerate simulation
    # radial_list = radial_list[(radial_list >= 4.5) & (radial_list <= 7.5)]

    # Define angle distribution
    #n_angles = config_particles["n_angles"]
    #theta_list = np.linspace(0, 90, n_angles + 2)[1:-1]
    N_particles = int(50000) #int(1e6)


    bunch_intensity = 2.2e11
    normal_emitt_x = 2.2e-6 #m*rad
    normal_emitt_y = 2.2e-6 #m*rad
    sigma_z = 7.5e-2
    particle_ref = xp.Particles(
                        mass0=xp.PROTON_MASS_EV, q0=1, energy0=7000e9)
    #config_mad['beam_config']['lhcb1']['beam_energy_tot'])
    gaussian_bunch = xp.generate_matched_gaussian_bunch(
            num_particles = N_particles, total_intensity_particles = bunch_intensity,
            nemitt_x = normal_emitt_x, nemitt_y=normal_emitt_y, sigma_z = sigma_z,
            particle_ref = particle_ref,
            line = collider['lhcb1'])
    # Define particle distribution as a cartesian product of the above
    particle_list = [
        (particle_id, x, y, px, py, zeta, delta)
        for particle_id, (x, y, px, py, zeta, delta) in enumerate(zip(gaussian_bunch.x, gaussian_bunch.y, gaussian_bunch.px, gaussian_bunch.py, gaussian_bunch.zeta, gaussian_bunch.delta))
    ]
    print('first',particle_list)
    # Split distribution into several chunks for parallelization
    n_split = config_particles['n_split']
    particle_list = np.array(np.array_split(particle_list, n_split))
    array_of_lists = np.array([arr.tolist() for arr in particle_list])
    particle_list = array_of_lists
    print('second',particle_list)

    # Return distribution
    return particle_list

def write_particle_distribution(particle_list):
    # Write distribution to parquet files
    distributions_folder = "particles_new"
    os.makedirs(distributions_folder, exist_ok=True)
    for idx_chunk, my_list in enumerate(particle_list):
        pd.DataFrame(
            my_list,
            columns=["particle_id", "x", "y", "px", "py", "zeta", "delta"],
        ).to_parquet(f"{distributions_folder}/{idx_chunk:02}.parquet")



# END NEW


# ==================================================================================================
# --- Function to build collider from mad model
# ==================================================================================================
def build_collider_from_mad(config_mad, context, sanity_checks=True):
    # Make mad environment
    xm.make_mad_environment(links=config_mad["links"])

    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")

    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    ost.build_sequence(mad_b1b2, mylhcbeam=1)
    ost.build_sequence(mad_b4, mylhcbeam=4)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    ost.apply_optics(mad_b1b2, optics_file=config_mad["optics_file"])

    if sanity_checks:
        mad_b1b2.use(sequence="lhcb1")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)
        mad_b1b2.use(sequence="lhcb2")
        mad_b1b2.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Apply optics (only for b4, just for check)
    ost.apply_optics(mad_b4, optics_file=config_mad["optics_file"])
    if sanity_checks:
        mad_b4.use(sequence="lhcb2")
        mad_b4.twiss()
        ost.check_madx_lattices(mad_b1b2)

    # Build xsuite collider
    collider = xlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config=config_mad["beam_config"],
        enable_imperfections=config_mad["enable_imperfections"],
        enable_knob_synthesis=config_mad["enable_knob_synthesis"],
        rename_coupling_knobs=config_mad["rename_coupling_knobs"],
        pars_for_imperfections=config_mad["pars_for_imperfections"],
        ver_lhc_run=config_mad["ver_lhc_run"],
        ver_hllhc_optics=config_mad["ver_hllhc_optics"],
    )
    collider.build_trackers(_context=context)

    if sanity_checks:
        collider["lhcb1"].twiss(method="4d")
        collider["lhcb2"].twiss(method="4d")
    # Return collider
    return collider


def activate_RF_and_twiss(collider, config_mad, context, sanity_checks=True):
    # Define a RF system (values are not so immportant as they're defined later)
    print("--- Now Computing Twiss assuming:")
    if config_mad["ver_hllhc_optics"] == 1.6:
        dic_rf = {"vrf400": 16.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.5}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    elif config_mad["ver_lhc_run"] == 3.0:
        dic_rf = {"vrf400": 12.0, "lagrf400.b1": 0.5, "lagrf400.b2": 0.0}
        for knob, val in dic_rf.items():
            print(f"    {knob} = {val}")
    print("---")

    # Rebuild tracker if needed
    try:
        collider.build_trackers(_context=context)
    except:
        print("Skipping rebuilding tracker")

    for knob, val in dic_rf.items():
        collider.vars[knob] = val

    if sanity_checks:
        for my_line in ["lhcb1", "lhcb2"]:
            ost.check_xsuite_lattices(collider[my_line])

    return collider


def clean():
    # Remove all the temporaty files created in the process of building collider
    os.remove("mad_collider.log")
    os.remove("mad_b4.log")
    shutil.rmtree("temp")
    os.unlink("errors")
    os.unlink("acc-models-lhc")


# ==================================================================================================
# --- Main function for building distribution and collider
# ==================================================================================================
def build_distr_and_collider(config_file="config.yaml"):
    # Get configuration
    configuration, config_particles, config_mad = load_configuration(config_file)

    # Get context
    context = get_context(configuration)

    # Get sanity checks flag
    sanity_checks = configuration["sanity_checks"]

    # Tag start of the job
    tree_maker_tagging(configuration, tag="started")

    
    

    # Build collider from mad model
    collider = build_collider_from_mad(config_mad, context, sanity_checks)

    # Twiss to ensure eveyrthing is ok
    collider = activate_RF_and_twiss(collider, config_mad, context, sanity_checks)
    
    # Build particle distribution
    particle_list = build_particle_distribution(config_mad, config_particles, collider)

    # Write particle distribution to file
    write_particle_distribution(particle_list)
    
    # Clean temporary files
    clean()

    # Save collider to json
    os.makedirs("collider", exist_ok=True)
    collider.to_json("collider/collider.json")

    # Tag end of the job
    tree_maker_tagging(configuration, tag="completed")


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    build_distr_and_collider()

 # %%
