from enum import Enum
from flowermd.base import Pack, Lattice, Simulation
from flowermd.library import EllipsoidForcefield, EllipsoidChain, PPS, OPLS_AA_PPS
from flowermd.utils import get_target_box_number_density, get_target_box_mass_density
from flowermd.utils.constraints import create_rigid_ellipsoid_chain
from time import sleep
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
import rowan
import sys
import unyt as u
import warnings

def translate_ellipsoid_by(ellipsoid, translation):
    '''
    Translate an ellipsoid by the given translation

    Example:
    ellipsoids = EllipsoidChain(num_mols=2, lpar=1.0, bead_mass=1.0, lengths=1)
    system = Pack(density=0.01*u.Unit("nm**-3"), molecules=ellipsoids)
    translate_ellipsoid_by(system.system.children[0].children[0], [1.0, 0.2, 0])
    '''

    for child in ellipsoid.children:
        child.pos += translation
    return ellipsoid

def rotation_matrix_x(val):
    return np.array([[1, 0,           0           ],
                     [0, np.cos(val), -np.sin(val)],
                     [0, np.sin(val), np.cos(val) ]])

def rotation_matrix_y(val):
    return np.array([[np.cos(val),  0, np.sin(val)],
                     [0,            1, 0          ],
                     [-np.sin(val), 0, np.cos(val)]])

def rotation_matrix_z(val):
    return np.array([[np.cos(val), -np.sin(val), 0],
                     [np.sin(val), np.cos(val),  0],
                     [0,           0,            1]])

def rotate_ellipsoid_by(ellipsoid, rotation):
    '''
    Rotate an ellipsoid by the given rotation

    Example:
    ellipsoids = EllipsoidChain(num_mols=2, lpar=1.0, bead_mass=1.0, lengths=1)
    system = Pack(density=0.01*u.Unit("nm**-3"), molecules=ellipsoids)
    rotate_ellipsoid_by(system.system.children[0].children[0], [numpy.pi/2, 0, numpy.pi])
    '''

    center_particle = ellipsoid.children[0]
    for child in ellipsoid.children:
        child_rel_to_center = child.pos - center_particle.pos

        # there's probably a more efficient way to pre-compute the multiplication
        # of each rotation matrix together into  but I don't feel like doing it rn
        child_rel_to_center = rotation_matrix_x(rotation[0]).dot(child_rel_to_center)
        child_rel_to_center = rotation_matrix_y(rotation[1]).dot(child_rel_to_center)
        child_rel_to_center = rotation_matrix_z(rotation[2]).dot(child_rel_to_center)

        child.pos = center_particle.pos + child_rel_to_center
    return ellipsoid

def ellipsoid_to_origin(ellipsoid):
    '''
    Move an ellipsoid such that it's center (X particle)
    is at (0, 0, 0), and is parallel with the X-axis

    Example:
    ellipsoids = EllipsoidChain(num_mols=2, lpar=1.0, bead_mass=1.0, lengths=1)
    system = Pack(density=0.01*u.Unit("nm**-3"), molecules=ellipsoids)
    ellipsoid_to_origin(system.system.children[0].children[0])
    '''
    center = ellipsoid.children[0].pos
    bond = ellipsoid.children[1].pos
    head = ellipsoid.children[2].pos

    lpar = np.linalg.norm(head - center)
    bond_from_center = np.linalg.norm(bond - center)

    ellipsoid.children[0].pos = np.array([0.0, 0.0, 0.0])
    ellipsoid.children[1].pos = np.array([bond_from_center, 0.0, 0.0])
    ellipsoid.children[2].pos = np.array([-lpar, 0.0, 0.0])
    ellipsoid.children[3].pos = np.array([lpar, 0.0, 0.0])

def ellipsoid_gsd(gsd_file, new_file, ellipsoid_types, lpar, lperp):
    """Add needed information to GSD file to visualize ellipsoids.

    Saves a new GSD file with lpar and lperp values populated
    for each particle. Ovito can be used to visualize the new GSD file.

    Parameters
    ----------
    gsd_file : str
        Path to the original GSD file containing trajectory information
    new_file : str
        Path and filename of the new GSD file
    ellipsoid_types : str or list of str
        The particle types (i.e. names) of particles to be drawn
        as ellipsoids.
    lpar : float
        Value of lpar of the ellipsoids
    lperp : float
        Value of lperp of the ellipsoids

    """
    with gsd.hoomd.open(new_file, "w") as new_t:
        with gsd.hoomd.open(gsd_file) as old_t:
            for snap in old_t:
                shape_dicts_list = []
                for ptype in snap.particles.types:
                    if ptype == ellipsoid_types or ptype in ellipsoid_types:
                        shapes_dict = {
                            "type": "Ellipsoid",
                            "a": lpar,
                            "b": lperp,
                            "c": lperp,
                        }
                    else:
                        shapes_dict = {"type": "Sphere", "diameter": 0.001}
                    shape_dicts_list.append(shapes_dict)
                snap.particles.type_shapes = shape_dicts_list
                snap.validate()
                new_t.append(snap)

def create_rigid_ellipsoid_chain(snapshot, orientations = None):
    """Create rigid bodies from a snapshot.

    This is designed to be used with flowerMD's built in library
    for simulating ellipsoidal chains.
    As a result, this will not work for setting up rigid bodies
    for other kinds of systems.

    See `flowermd.library.polymer.EllipsoidChain` and
    `flowermd.library.forcefields.EllipsoidForcefield`.

    Parameters
    ----------
    snapshot : gsd.hoomd.Snapshot; required
        The snapshot of the system.
        Pass in `flowermd.base.System.hoomd_snapshot()`.

    Returns
    -------
    rigid_frame : gsd.hoomd.Frame
        The snapshot of the rigid bodies.
    rigid_constrain : hoomd.md.constrain.Rigid
        The rigid body constrain object.

    """
    bead_len = 4  # Number of particles belonging to 1 rigid body
    typeids = snapshot.particles.typeid.reshape(-1, bead_len)
    matches = np.where((typeids == typeids))
    rigid_const_idx = (matches[0] * bead_len + matches[1]).reshape(-1, bead_len)
    n_rigid = rigid_const_idx.shape[0]  # number of ellipsoid monomers

    rigid_masses = []
    rigid_pos = []
    rigid_moi = []
    # Find the mass, position and MOI for reach rigid center
    for idx in rigid_const_idx:
        mass = np.sum(np.array(snapshot.particles.mass)[idx])
        pos = snapshot.particles.position[idx][0]
        rigid_masses.append(mass)
        rigid_pos.append(pos)
        rigid_moi.append([0, 2, 2])

    rigid_frame = gsd.hoomd.Frame()
    rigid_frame.particles.types = ["R"] + snapshot.particles.types
    rigid_frame.particles.N = n_rigid + snapshot.particles.N
    rigid_frame.particles.typeid = np.concatenate(
        (([0] * n_rigid), snapshot.particles.typeid + 1)
    )
    rigid_frame.particles.mass = np.concatenate(
        (rigid_masses, snapshot.particles.mass)
    )
    rigid_frame.particles.position = np.concatenate(
        (rigid_pos, snapshot.particles.position)
    )
    rigid_frame.particles.moment_inertia = np.concatenate(
        (rigid_moi, np.zeros((snapshot.particles.N, 3)))
    )
    
    if orientations is not None:
        # initialize orientation to correct dimensions
        rigid_frame.particles.orientation = [orientations[0]]

        # rigid body orientations
        if len(orientations) > 1:
            for orientation in orientations[1:]:
                rigid_frame.particles.orientation = np.vstack((rigid_frame.particles.orientation, orientation))

        # individual particle orientations
        for orientation in orientations:
            for i in range(4): # one orientation per particle that matches rigid body orientation
                rigid_frame.particles.orientation = np.vstack((rigid_frame.particles.orientation, orientation))

        print(f"ORIENTATIONS: {rigid_frame.particles.orientation}")
    else:
        rigid_frame.particles.orientation = [[1, 0, 0, 0]] * (
            n_rigid + snapshot.particles.N
        )
        
    rigid_frame.particles.body = np.concatenate(
        (
            np.arange(n_rigid),
            np.arange(n_rigid).repeat(rigid_const_idx.shape[1]),
        )
    )
    rigid_frame.configuration.box = snapshot.configuration.box

    # set up bonds
    if snapshot.bonds.N > 0:
        rigid_frame.bonds.N = snapshot.bonds.N
        rigid_frame.bonds.types = snapshot.bonds.types
        rigid_frame.bonds.typeid = snapshot.bonds.typeid
        rigid_frame.bonds.group = [
            list(np.add(g, n_rigid)) for g in snapshot.bonds.group
        ]
    # set up angles
    if snapshot.angles.N > 0:
        rigid_frame.angles.N = snapshot.angles.N
        rigid_frame.angles.types = snapshot.angles.types
        rigid_frame.angles.typeid = snapshot.angles.typeid
        rigid_frame.angles.group = [
            list(np.add(g, n_rigid)) for g in snapshot.angles.group
        ]

    # find local coordinates of the particles in the first rigid body
    # only need to find the local coordinates for the first rigid body
    local_coords = (
        snapshot.particles.position[rigid_const_idx[0]] - rigid_pos[0]
    )

    rigid_constrain = hoomd.md.constrain.Rigid()
    rigid_constrain.body["R"] = {
        "constituent_types": ["X", "A", "T", "T"],
        "positions": local_coords,
        "orientations": [[1, 0, 0, 0]] * len(local_coords),
    }
    return rigid_frame, rigid_constrain

# Credit to marjanalbooyeh for writing this
# https://github.com/cmelab/ellipsoid_orientations/blob/main/src/quaternions.py
def rotate_quaternion(q, theta, axis):
    cos_theta_2 = np.cos(theta / 2)
    sin_theta_2 = np.sin(theta / 2)

    if axis == 'x':
        r = [cos_theta_2, sin_theta_2, 0, 0]
    elif axis == 'y':
        r = [cos_theta_2, 0, sin_theta_2, 0]
    elif axis == 'z':
        r = [cos_theta_2, 0, 0, sin_theta_2]
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    q0, q1, q2, q3 = q
    r0, r1, r2, r3 = r

    q_prime = [
        r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3,
        r0 * q1 + r1 * q0 + r2 * q3 - r3 * q2,
        r0 * q2 - r1 * q3 + r2 * q0 + r3 * q1,
        r0 * q3 + r1 * q2 - r2 * q1 + r3 * q0
    ]

    return rowan.normalize(q_prime)                
