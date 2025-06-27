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
import util

warnings.filterwarnings('ignore')

MAX_DIST = 1e-23
MIN_DIST = 0.0
RESOLUTION = int(1e1)
TIME_STRING = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR = 'output/' + TIME_STRING + '/'
GSD_FILE_NAME = OUTPUT_DIR + 'trajectory.gsd'
LOG_FILE_NAME = OUTPUT_DIR + 'trajectory.gsd'

for i in range(RESOLUTION):
    ellipsoid = EllipsoidChain(num_mols=2, lpar=1.0, bead_mass=1.0, lengths=1)
    system = Pack(density=0.0001*u.Unit("nm**-3"), molecules=ellipsoid)

    ellipsoid_to_origin(system.system.children[0].children[0])
    ellipsoid_to_origin(system.system.children[1].children[0])
    translate_ellipsoid_by(system.system.children[1].children[0], [0.0, ((MAX_DIST-MIN_DIST)/(RESOLUTION))*i+MIN_DIST, 0.0])

    ff = EllipsoidForcefield(
        epsilon=1.0,
        lpar=1.0,
        lperp=0.5,
        r_cut=3.0,
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
        gsd_file_name=f'{i}-trajectory.gsd',
        log_write_freq=1,
        log_file_name=f'{i}-log.txt',
        dt=0.001
    )

    ellipsoid_sim.run_NVT(n_steps=5, kT=1.0, tau_kt=1.0, thermalize_particles=False)

    ellipsoid_sim.flush_writers()

    ellipsoid_gsd(
        gsd_file=f'{i}-trajectory.gsd',
        new_file=f'{i}-ovito-trajectory.gsd',
        ellipsoid_types='R',
        lpar=1.0,
        lperp=0.5,
    )

potential = np.array([])
radius = np.array([])
for i in range(RESOLUTION):
    print('hi')
    log = np.genfromtxt(f'{i}-log.txt', names=True)
    potential_entry = log["mdcomputeThermodynamicQuantitiespotential_energy"][-1]
    potential = np.append(potential, potential_entry)
    radius_entry = ((MAX_DIST-MIN_DIST)/(RESOLUTION))*i+MIN_DIST
    radius = np.append(radius, radius_entry)
    print(f"potential_entry: {potential_entry}, radius_entry: {radius_entry}")

plt.plot(radius, potential/2)
plt.title('Potential Energy Per Particle vs. Radius')
plt.xlabel('Radius')
plt.ylabel('Potential Energy Per Particle')
plt.savefig('potential-energy.png')
plt.close()
