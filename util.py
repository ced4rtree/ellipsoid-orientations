import numpy as np
import gsd.hoomd

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
    head = ellipsoid.children[2].pos
    bond = ellipsoid.children[3].pos

    lpar = np.linalg.norm(head - center)
    bond_from_center = np.linalg.norm(bond - center)

    ellipsoid.children[0].pos = np.array([0.0, 0.0, 0.0])
    ellipsoid.children[1].pos = np.array([-lpar, 0.0, 0.0])
    ellipsoid.children[2].pos = np.array([lpar, 0.0, 0.0])
    ellipsoid.children[3].pos = np.array([bond_from_center, 0.0, 0.0])

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

