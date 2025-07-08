#!/usr/bin/env python

# compilation command on NixOS:
# bash -c 'conda-shell -c "conda activate ellipsoids && python ellipsoids.py"'

from flowermd.base import Pack, Lattice, Simulation
from flowermd.library import EllipsoidForcefield, EllipsoidChain, PPS, OPLS_AA_PPS
from flowermd.utils import get_target_box_number_density, get_target_box_mass_density
from time import sleep
from typing import Tuple, List
from util import *
import argparse
import datetime
import gsd
import gsd.hoomd
import hoomd
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import unyt as u
import warnings

warnings.filterwarnings('ignore')

RESOLUTION = 200
PROGRESS_BAR_WIDTH = 20

R_CUT = 10.0
EPSILON = 1.0
LPAR=1.0
LPERP=0.5

MAX_DIST = 4.0
MIN_DIST = 0.1
TIME_STRING = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_DIR = 'output/' + TIME_STRING + '/'

try:
    os.makedirs(OUTPUT_DIR)
except OSError as error:
    print('failed to create directory ' + OUTPUT_DIR)
    exit(1)

# ellipsoids oriented like ||
parallel_potential = np.array([])
parallel_radius = np.array([])

# ellipsoids oriented like <><>
parallel_long_potential = np.array([])
parallel_long_radius = np.array([])

# ellipsoids oriented like |<>
perpendicular_potential = np.array([])
perpendicular_radius = np.array([])

class Direction(Enum):
    X = 0
    Y = 1
    Z = 2

def simulate(
        direction: Direction,
        name: str,
        rotation: np.ndarray = None,
        orientations: np.ndarray = None,
) -> Tuple[List[float], List[float]]:
    '''Do many simulations of two ellipsoids, where the first is located at (0,
        0, 0) and parallel to the x-axis, and the other is placed at RESOLUTION
        points between MIN_DIST and MAX_DIST away from the first along the
        specified direction, and rotated around the x, y, and z axis relative to
        its center.

    Parameters
    ----------
    rotation: np.ndarray, An array of rotation values in radians, formatted like
    [x, y, z]

    direction: Direction, The direction for the simulation to sample potentials
    in
    name: str, The name that will be used for identifying logs

    Return Value
    ------------
    Returns an array of input radii and corresponding array of output potential
    values

    '''

    potential = [None] * (RESOLUTION)
    radius = [None] * (RESOLUTION)

    for i in range(RESOLUTION):
        gsd_file_name = OUTPUT_DIR + f'{name}-{i}.gsd'
        log_file_name = OUTPUT_DIR + f'{name}-{i}.txt'
        
        ellipsoid = EllipsoidChain(num_mols=2, lpar=LPAR, bead_mass=1.0, lengths=1)
        system = Pack(density=0.1*u.Unit("nm**-3"), molecules=ellipsoid)

        # scaled to be MIN_DIST and MAX_DIST inclusive while using only
        # RESOLUTION many steps
        max_dist_scaled = MAX_DIST
        dist_step = ((MAX_DIST-MIN_DIST)/RESOLUTION)
        max_dist_scaled += dist_step
        ellipsoid_dist = ((max_dist_scaled-MIN_DIST)/(RESOLUTION))*i+MIN_DIST
        
        ellipsoid_to_origin(system.system.children[0].children[0])
        ellipsoid_to_origin(system.system.children[1].children[0])
        translation = [0.0, 0.0, 0.0]
        translation[direction.value] = ellipsoid_dist
        translate_ellipsoid_by(system.system.children[1].children[0], translation)
        if rotation is not None:
            rotate_ellipsoid_by(system.system.children[1].children[0], rotation)
        system.gmso_system = system._convert_to_gmso()
        system._hoomd_snapshot = system._create_hoomd_snapshot()

        print(f"System positions before: {system.system.children[1].children[0].children}\n")
        
        ff = EllipsoidForcefield(
            epsilon=EPSILON,
            lpar=LPAR,
            lperp=LPERP,
            r_cut=R_CUT,
            bond_r0=0.1,
            bond_k=100
        )
        rigid_frame, rigid_constraint = create_rigid_ellipsoid_chain(
            system.hoomd_snapshot,
            orientations=orientations
        )

        print(f"Rigid frame positions after: {rigid_frame.particles.position}\n")
        
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
            lpar=LPAR,
            lperp=LPERP,
        )
        
        log = np.genfromtxt(log_file_name, names=True)
        potential_entry = log["mdcomputeThermodynamicQuantitiespotential_energy"]

        radius[i] = ellipsoid_dist
        potential[i] = potential_entry
        
        # print progress bar
        # https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage
        print(f"potential_entry: {potential_entry}\nradius_entry: {ellipsoid_dist}\nlog file: {log_file_name}")
        sys.stdout.write('\r')
        sys.stdout.write(f"[%-{PROGRESS_BAR_WIDTH}s] %d%%\n\n" % ('='*int(i/(RESOLUTION/PROGRESS_BAR_WIDTH)), 100/RESOLUTION*i))
        sys.stdout.flush()

    return radius, potential

if __name__ == '__main__':
    # ||
    parallel = simulate(
        Direction.Y,
        "parallel",
    )
    parallel_radius = np.append(parallel_radius, parallel[0])
    parallel_potential = np.append(parallel_potential, parallel[1])
    
    # <><>
    parallel_long = simulate(
        Direction.Z,
        "parallel_long",
    )
    parallel_long_radius = np.append(parallel_long_radius, parallel_long[0])
    parallel_long_potential = np.append(parallel_long_potential, parallel_long[1])
    
    # |<>
    perpendicular = simulate(
        Direction.Y,
        "perpendicular",
        rotation = np.array([0, 0, np.pi/2]),
        orientations = np.array([[1, 0, 0, 0], rotate_quaternion([1, 0, 0, 0], np.pi/2, 'x')]),
    )
    perpendicular_radius = np.append(perpendicular_radius, perpendicular[0])
    perpendicular_potential = np.append(perpendicular_potential, perpendicular[1])

    print(f"parallel radius && parallel_long radius: {parallel_radius == parallel_long_radius}")
    print(f"perpendicular radius && parallel_long radius: {perpendicular_radius == parallel_long_radius}")
    print(f"perpendicular radius && parallel radius: {perpendicular_radius == parallel_radius}")
    print(f"parallel potential && parallel_long potential: {parallel_potential == parallel_long_potential}")
    print(f"perpendicular potential && parallel_long potential: {perpendicular_potential == parallel_long_potential}")
    print(f"perpendicular potential && parallel potential: {perpendicular_potential == parallel_potential}")
    
    plt.plot(parallel_radius, parallel_potential/2, label='Parallel (||)')
    plt.plot(parallel_long_radius, parallel_long_potential/2, label='Parallel Long (<><>)')
    plt.plot(perpendicular_radius, perpendicular_potential/2, label='Perpendicular (|<>)')
    plt.title('Potential Energy Per Particle vs. Center-To-Center Distance')
    plt.xlabel('Center-To-Center Distance')
    plt.ylabel('Potential Energy Per Particle')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}potential-energy-per.png')
    plt.close()
    
    plt.plot(parallel_radius, parallel_potential, label='Parallel (||)')
    plt.plot(parallel_long_radius, parallel_long_potential, label='Parallel Long (<><>)')
    plt.plot(perpendicular_radius, perpendicular_potential, label='Perpendicular (|<>)')
    plt.title('Potential Energy vs. Center-To-Center Distance')
    plt.xlabel('Center-To-Center Distance')
    plt.ylabel('Potential Energy')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}potential-energy.png')
    plt.close()
    
    plt.plot(parallel_radius, parallel_potential, label='Parallel (||)')
    plt.plot(parallel_long_radius, parallel_long_potential, label='Parallel Long (<><>)')
    plt.plot(perpendicular_radius, perpendicular_potential, label='Perpendicular (|<>)')
    plt.title('Potential Energy vs. Center-To-Center Distance')
    plt.xlabel('Center-To-Center Distance')
    plt.ylabel('Potential Energy')
    plt.ylim(-EPSILON-0.5, 0.5)
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}potential-energy-limited.png')
    plt.close()

    plt.plot(parallel_radius, parallel_potential/2, label='Parallel (||)')
    plt.plot(parallel_long_radius, parallel_long_potential/2, label='Parallel Long (<><>)')
    plt.plot(perpendicular_radius, perpendicular_potential/2, label='Perpendicular (|<>)')
    plt.title('Potential Energy Per Particle vs. Center-To-Center Distance')
    plt.xlabel('Center-To-Center Distance')
    plt.ylabel('Potential Energy Per Particle')
    plt.ylim(-EPSILON-0.5, 0.5)
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}potential-energy-limited-per.png')
    plt.close()
