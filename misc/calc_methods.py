import numpy as np

ref_system = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def get_rotation_matrix(ref_system, rotated_system):
    """
    Generate a 3x3 rotation matrix using orthogonal unit vectors.
    :param ref_system: Reference coordinate system [x, y, z]
    :param rotated_system: Target coordinate system [xr, yr, zr]
    :return: 3x3 Rotation Matrix
    """
    r00 = np.dot(rotated_system[0], ref_system[0])
    r10 = np.dot(rotated_system[0], ref_system[1])
    r20 = np.dot(rotated_system[0], ref_system[2])
    r01 = np.dot(rotated_system[1], ref_system[0])
    r11 = np.dot(rotated_system[1], ref_system[1])
    r21 = np.dot(rotated_system[1], ref_system[2])
    r02 = np.dot(rotated_system[2], ref_system[0])
    r12 = np.dot(rotated_system[2], ref_system[1])
    r22 = np.dot(rotated_system[2], ref_system[2])
    matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return matrix


def get_transformation(s_system, scenter, t_system, tcenter):
    """
    Compute the 4x4 transformation matrix from one coordinate system to another.
    :param np.array s_system: 3x3 Matrix of the source coordinate system
    :param np.array scenter: Center of the source coordinate system [x, y, z]
    :param np.array t_system: 3x3 Matrix of the target coordinate system
    :param np.array tcenter: Center of the target coordinate system [x, y, z]
    :return np.array: 4x4 Transformation Matrix
    """
    world = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rot_world2to = get_rotation_matrix(world, t_system)
    rot = get_rotation_matrix(s_system, t_system)

    tx = scenter[0] - tcenter[0]
    ty = scenter[1] - tcenter[1]
    tz = scenter[2] - tcenter[2]

    # Build the transformation matrix
    Transf = [
        [rot[0][0], rot[0][1], rot[0][2], rot_world2to[0][0] * tx + rot_world2to[0][1] * ty + rot_world2to[0][2] * tz],
        [rot[1][0], rot[1][1], rot[1][2], rot_world2to[1][0] * tx + rot_world2to[1][1] * ty + rot_world2to[1][2] * tz],
        [rot[2][0], rot[2][1], rot[2][2], rot_world2to[2][0] * tx + rot_world2to[2][1] * ty + rot_world2to[2][2] * tz],
        [0, 0, 0, 1]]
    return np.array(Transf)


def get_system_from_4_clusters(cluster):
    """
    Selects a consistent coordinate system from four markers.
    The z-axis points perpendicular to the approximate plane formed by the markers, pointing 'upwards'.
    :param cluster: List of four markers [marker0, marker1, marker2, marker3]
    :return: coordinate system [versor0, versor1, versor2], center
    """
    assert len(cluster) == 4, "Cluster must contain exactly 4 markers."

    # Se toma centro de los clusters como centro del sistema
    center = np.mean(cluster, axis=0)


    # Calculo de vectores
    vec1 = cluster[1] - cluster[0]
    vec2 = cluster[2] - cluster[0]

    # Vector normal al plano formado por vec1 y vec2
    normal = np.cross(vec1, vec2)
    normal = normal / np.linalg.norm(normal)  # Normalizacion

    # Asegurar que el normal apunta hacia positivos
    if normal[2] < 0:
        normal = -normal
    z_axis = normal

    # Eje x como la proyeccion de vec 1 en el plano
    x_axis = vec1 - np.dot(vec1, z_axis) * z_axis
    x_axis = x_axis / np.linalg.norm(x_axis)  # Normalizacion

    # Calculo eje Y como el producto de Z y X.
    y_axis = np.cross(z_axis, x_axis)

    return np.transpose(np.array([x_axis, y_axis, z_axis])), center


def get_system_from_3_clusters(cluster):
    """
    Select always the same cluster 0, 1 and 2, and calculates a new coordinate from the marker positions.
    To select the same cluster as 0, it calculate the distances between them.
    :param cluster: [marker0, marker1, marker2]
    :return: coordinate system [versor0, versor1, versor2]
    """
    center = None
    d1 = np.linalg.norm(cluster[0] - cluster[1])
    d2 = np.linalg.norm(cluster[0] - cluster[2])
    d3 = np.linalg.norm(cluster[1] - cluster[2])
    if d1 < d2 and d1 < d3:
        if d2 < d3:
            A = cluster[1]
            B = cluster[2]
            C = cluster[0]
            center = C
            V1 = (A - C) / np.linalg.norm(A - C)
            V2 = np.cross(V1, (B - C)) / np.linalg.norm(np.cross(V1, (B - C)))
            V3 = np.cross(V1, V2)
        else:
            A = cluster[0]
            B = cluster[2]
            C = cluster[1]
            center = A
            V1 = (B - A) / np.linalg.norm(B - A)
            V2 = np.cross(V1, (C - A)) / np.linalg.norm(np.cross(V1, (C - A)))
            V3 = np.cross(V1, V2)
    elif d2 < d1 and d2 < d3:
        if d1 < d3:
            A = cluster[0]
            B = cluster[1]
            C = cluster[2]
            center = A
            V1 = (B - A) / np.linalg.norm(B - A)
            V2 = np.cross(V1, (C - A)) / np.linalg.norm(np.cross(V1, (C - A)))
            V3 = np.cross(V1, V2)
        else:
            A = cluster[2]
            B = cluster[1]
            C = cluster[0]
            center = C
            V1 = (A - C) / np.linalg.norm(A - C)
            V2 = np.cross(V1, (B - C)) / np.linalg.norm(np.cross(V1, (B - C)))
            V3 = np.cross(V1, V2)
    else:
        if d1 < d2:
            A = cluster[2]
            B = cluster[0]
            C = cluster[1]
            center = B
            V1 = (A - B) / np.linalg.norm(A - B)
            V2 = np.cross(V1, (C - B)) / np.linalg.norm(np.cross(V1, (C - B)))
            V3 = np.cross(V1, V2)
        else:
            A = cluster[1]
            B = cluster[0]
            C = cluster[2]
            center = B
            V1 = (A - B) / np.linalg.norm(A - B)
            V2 = np.cross(V1, (C - B)) / np.linalg.norm(np.cross(V1, (C - B)))
            V3 = np.cross(V1, V2)

    return np.transpose(np.array([V1, V2, V3])), np.array(center)