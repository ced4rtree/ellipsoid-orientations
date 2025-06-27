#!/usr/bin/env python
from flowermd.base import Pack, Lattice, Simulation
from flowermd.library import EllipsoidForcefield, EllipsoidChain, PPS, OPLS_AA_PPS
from flowermd.utils import get_target_box_number_density, get_target_box_mass_density
from flowermd.utils.constraints import create_rigid_ellipsoid_chain
import argparse
import datetime
import gsd
import gsd.hoomd
import hoomd
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import unyt as u
import warnings
from util import *

warnings.filterwarnings('ignore')

R_CUT = 3.0
EPSILON = 1.0

MAX_DIST = R_CUT*1.1
MIN_DIST = 0.1
RESOLUTION = int(1e2)
TIME_STRING = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR = 'output/' + TIME_STRING + '/'

try:
    os.makedirs(OUTPUT_DIR)
except OSError as error:
    print('failed to create directory ' + OUTPUT_DIR)
    exit(1)

potential = np.array([])
radius = np.array([])

for i in range(RESOLUTION+1): # +1 for inclusive range
    gsd_file_name = OUTPUT_DIR + f'trajectory-{i}.gsd'
    log_file_name = OUTPUT_DIR + f'log-{i}.txt'

    ellipsoid = EllipsoidChain(num_mols=2, lpar=1.0, bead_mass=1.0, lengths=1)
    system = Pack(density=0.00001*u.Unit("nm**-3"), molecules=ellipsoid)

    ellipsoid_dist = ((MAX_DIST-MIN_DIST)/(RESOLUTION))*i+MIN_DIST

    ellipsoid_to_origin(system.system.children[0].children[0])
    ellipsoid_to_origin(system.system.children[1].children[0])
    translate_ellipsoid_by(system.system.children[1].children[0], [0.0, ellipsoid_dist, 0.0])
    system.gmso_system = system._convert_to_gmso()

    ff = EllipsoidForcefield(
        epsilon=EPSILON,
        lpar=1.0,
        lperp=0.5,
        r_cut=R_CUT,
        bond_r0=0.1,
        bond_k=100
    )
    rigid_frame, rigid_constraint = create_rigid_ellipsoid_chain(
        system.hoomd_snapshot
    )

    ellipsoid_sim = Simulation(
        initial_state=rigid_frame,
        forcefield=ff.hoomd_forces,
        constraint=rigid_constraint,
        gsd_write_freq=1,
        gsd_file_name=gsd_file_name,
        log_write_freq=1,
        log_file_name=log_file_name,
        dt=0.001
    )

    ellipsoid_sim.run_NVT(n_steps=0, kT=1.0, tau_kt=1.0, thermalize_particles=False)

    ellipsoid_sim.flush_writers()

    ellipsoid_gsd(
        gsd_file=gsd_file_name,
        new_file=gsd_file_name.replace('.gsd', '-ovito.gsd'),
        ellipsoid_types='R',
        lpar=1.0,
        lperp=0.5,
    )

    log = np.genfromtxt(log_file_name, names=True)
    potential_entry = log["mdcomputeThermodynamicQuantitiespotential_energy"]
    potential = np.append(potential, potential_entry)
    radius = np.append(radius, ellipsoid_dist)
    print(f"potential_entry: {potential_entry}, radius_entry: {ellipsoid_dist}")

plt.plot(radius, potential/2)
plt.title('Potential Energy Per Particle vs. Radius')
plt.xlabel('Radius')
plt.ylabel('Potential Energy Per Particle')
plt.savefig(f'{OUTPUT_DIR}potential-energy.png')
plt.close()

plt.plot(radius, potential/2)
plt.title('Potential Energy Per Particle vs. Radius')
plt.xlabel('Radius')
plt.ylabel('Potential Energy Per Particle')
plt.ylim(-EPSILON-1, 2)
plt.savefig(f'{OUTPUT_DIR}potential-energy-limited.png')
plt.close()
