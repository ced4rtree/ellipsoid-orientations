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
import mbuild as mb
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
        orientation: np.ndarray = None,
) -> Tuple[List[float], List[float]]:
    '''Do many simulations of two ellipsoids, where the first is located at (0,
        0, 0) and parallel to the x-axis, and the other is placed at RESOLUTION
        points between MIN_DIST and MAX_DIST away from the first along the
        specified direction, and rotated around the x, y, and z axis relative to
        its center.

    Parameters
    ----------
    direction: Direction, The direction for the simulation to sample potentials
    in

    name: str, The name that will be used for identifying logs

    orientation: np.ndarray[float, float, float, float], Quaternion representing
    the orientation of the second ellipsoid

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
        
        # scaled to be MIN_DIST and MAX_DIST inclusive while using only
        # RESOLUTION many steps
        max_dist_scaled = MAX_DIST
        dist_step = ((MAX_DIST-MIN_DIST)/RESOLUTION)
        max_dist_scaled += dist_step
        ellipsoid_dist = ((max_dist_scaled-MIN_DIST)/(RESOLUTION))*i+MIN_DIST

        # create system and molecules
        translation = [0.0, 0.0, 0.0]
        translation[direction.value] = ellipsoid_dist
        box = mb.Compound()
        ellipsoid_one = mb.Compound(name="A", pos=(0, 0, 0), mass=1.0)
        ellipsoid_two = mb.Compound(name="A", pos=translation, mass=1.0)
        # if rotation is not None:
        #     ellipsoid_two.rotate(theta=rotation, around=ellipsoid_two.center)
        
        # dummy molecules to expand the box
        box.add(ellipsoid_one)
        box.add(ellipsoid_two)
        snapshot = mb.conversion.to_hoomdsnapshot(box)
        snapshot.configuration.box = [100, 100, 100, 0, 0, 0]
        
        # add Gay-Berne forcefield to the integrator
        nlist = hoomd.md.nlist.Cell(buffer=2.0, exclusions=["body"])
        gb = hoomd.md.pair.aniso.GayBerne(nlist=nlist, default_r_cut=R_CUT)
        gb.params[("A", "A")] = dict(
            epsilon=1.0, lperp=0.5, lpar=1.0
        )
        
        integrator = hoomd.md.Integrator(dt=0.001)
        integrator.forces.append(gb)
        
        # add NVT method to integrator
        nvt = hoomd.md.methods.ConstantVolume(
            filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5)
        )
        integrator.methods.append(nvt)
        
        simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
        simulation.create_state_from_snapshot(snapshot)
        simulation.operations.integrator = integrator
        
        # compute thermodynamic properties
        thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
            filter=hoomd.filter.All()
        )
        simulation.operations.computes.append(thermodynamic_properties)

        with simulation._state.cpu_local_snapshot as data:
            if orientation is not None:
                data.particles.orientation[1] = orientation

        simulation.run(0)

        # flush GSD writer
        if os.path.exists(gsd_file_name):
            os.remove(gsd_file_name)
        hoomd.write.GSD.write(state=simulation.state, filename=gsd_file_name, mode="xb")
        ellipsoid_gsd(
            gsd_file=gsd_file_name,
            new_file=gsd_file_name.replace('.gsd', '-ovito.gsd'),
            ellipsoid_types='A',
            lpar=LPAR,
            lperp=LPERP,
        )

        # print progress bar
        # https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage
        print(f"potential_entry: {thermodynamic_properties.potential_energy}\nradius_entry: {ellipsoid_dist}\nlog file: {log_file_name}")
        sys.stdout.write('\r')
        sys.stdout.write(f"[%-{PROGRESS_BAR_WIDTH}s] %d%%\n\n" % ('='*int(i/(RESOLUTION/PROGRESS_BAR_WIDTH)), 100/RESOLUTION*i))
        sys.stdout.flush()

        radius[i] = ellipsoid_dist
        potential[i] = thermodynamic_properties.potential_energy

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
        orientation = rotate_quaternion([1, 0, 0, 0], np.pi/2, 'x'),
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
    plt.title('Potential Energy Per Particle vs. Radius')
    plt.xlabel('Radius')
    plt.ylabel('Potential Energy Per Particle')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}potential-energy-per.png')
    plt.close()
    
    plt.plot(parallel_radius, parallel_potential, label='Parallel (||)')
    plt.plot(parallel_long_radius, parallel_long_potential, label='Parallel Long (<><>)')
    plt.plot(perpendicular_radius, perpendicular_potential, label='Perpendicular (|<>)')
    plt.title('Potential Energy vs. Radius')
    plt.xlabel('Radius')
    plt.ylabel('Potential Energy')
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}potential-energy.png')
    plt.close()
    
    plt.plot(parallel_radius, parallel_potential, label='Parallel (||)')
    plt.plot(parallel_long_radius, parallel_long_potential, label='Parallel Long (<><>)')
    plt.plot(perpendicular_radius, perpendicular_potential, label='Perpendicular (|<>)')
    plt.title('Potential Energy vs. Radius')
    plt.xlabel('Radius')
    plt.ylabel('Potential Energy')
    plt.ylim(-EPSILON-0.5, 0.5)
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}potential-energy-limited.png')
    plt.close()

    plt.plot(parallel_radius, parallel_potential/2, label='Parallel (||)')
    plt.plot(parallel_long_radius, parallel_long_potential/2, label='Parallel Long (<><>)')
    plt.plot(perpendicular_radius, perpendicular_potential/2, label='Perpendicular (|<>)')
    plt.title('Potential Energy Per Particle vs. Radius')
    plt.xlabel('Radius')
    plt.ylabel('Potential Energy Per Particle')
    plt.ylim(-EPSILON-0.5, 0.5)
    plt.legend()
    plt.savefig(f'{OUTPUT_DIR}potential-energy-limited-per.png')
    plt.close()
