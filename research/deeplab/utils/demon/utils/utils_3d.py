import cv2
import numpy as np
import math
import cython_util as cut
import pdb


def intrinsic_vec_to_mat(intrinsic, shape=[1,1]):

    K = np.zeros((3,3), dtype=np.float32)
    K[0, 0] = intrinsic[0] * shape[1]
    K[1, 1] = intrinsic[1] * shape[0]
    K[0, 2] = intrinsic[2] * shape[1]
    K[1, 2] = intrinsic[3] * shape[0]
    K[2, 2] = 1.0

    return K


def intrinsic_mat_to_vec(K):
    return np.array([K[0, 0], K[1, 1],
                     K[0, 2], K[1, 2]])


def down_sample_depth(depth, method='uniform', percent=0.01, K=None):
    """Down sample a depth image to simulate sparse sample case
       Supporting two types of samples:
           uniform distribution and a sweeping line

        Input: depth
        Ouput: depth_sample [N x 3] numpy matrix with sampled depth value

        K: normalized camera matrix [fx, fy, ux, uy]
    """

    if K is None:
        K = [1, 1, 0.5, 0.5]

    height, width = depth.shape[0], depth.shape[1]
    x3d = depth2xyz(depth, K, homo=False)
    valid = x3d[:, 2] > 0.05
    x3d = x3d[valid, :]

    if method == 'uniform':
        # sample uniformly for both x and y
        num = int(height * width * percent)
        sample = np.random.randint(x3d.shape[0],
                                   size=num)
        depth_sample = x3d[sample, :]

    if method == 'floor_plan':
        # sample in 3D
        height = 0.1
        idx = np.abs(x3d[:, 1] - 0.1) < 0.01
        x3d = x3d[idx, :]
        depth_sample = np.dot(K, x3d.transpose())
        depth_sample[0:2, :] /= depth_sample[2, :]

    return depth_sample


def inverse_depth(depth):
    """ calculate inverse of depth value
    """
    depth_inv = depth.copy()
    depth_inv[depth > 0] = 1. / depth_inv[depth > 0]
    return depth_inv


def depth2disp(depth, focal_len, base):
    assert np.ndim(depth) == 2
    depth_idx = depth > 0
    disp = np.zeros(depth.shape)
    disp[depth_idx] = focal_len * base / depth[depth_idx]
    return disp


def euler_angles_to_rotation_matrix(angle, is_dir=False):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
        is_dir: whether just use the 2d direction on a map
    """
    roll, pitch, yaw = angle[0], angle[1], angle[2]

    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]])

    pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]])

    yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]])

    R = yawMatrix * pitchMatrix * rollMatrix
    R = np.array(R)

    if is_dir:
        R = R[:, 2]

    return R


def euler_angles_to_quaternions(angle):
    """Convert euler angels to quaternions.
    Input:
        angle: [roll, pitch, yaw]
    """
    in_dim = np.ndim(angle)
    if in_dim == 1:
        angle = angle[None, :]

    n = angle.shape[0]
    roll, pitch, yaw = angle[:, 0], angle[:, 1], angle[:, 2]
    q = np.zeros((n, 4))

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q[:, 0] = cy * cr * cp + sy * sr * sp
    q[:, 1] = cy * sr * cp - sy * cr * sp
    q[:, 2] = cy * cr * sp + sy * sr * cp
    q[:, 3] = sy * cr * cp - cy * sr * sp

    return q[0] if in_dim == 1 else q


def quater_to_rot_mat(qu, is_dir=False):
    R = np.zeros((3,3))
    qw, qx, qy, qz = qu[0], qu[1], qu[2], qu[3]

    R[0, 0] = 1 - 2*qy*qy - 2*qz*qz
    R[0, 1] = 2*qx*qy - 2*qz*qw
    R[0, 2] = 2*qx*qz + 2*qy*qw

    R[1, 0] = 2*qx*qy + 2*qz*qw
    R[1, 1] = 1 - 2*qx*qx - 2*qz*qz
    R[1, 2] = 2*qy*qz - 2*qx*qw

    R[2, 0] = 2*qx*qz - 2*qy*qw
    R[2, 1] = 2*qy*qz + 2*qx*qw
    R[2, 2] = 1 - 2*qx*qx - 2*qy*qy

    if is_dir:
        R = R[:, 2]

    return R


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_matrix_to_euler_angles(R, check=True):
    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    if check:
        assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])

    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def depth2xyz(depth, K, homo=True, flat=True):
    """Convert a depth map to 3D point using camera intrinsic matrix
    """

    assert K[2] < 1. and K[3] < 1., "intrinsic must be normalized."
    depth = np.float32(depth)
    height, width = depth.shape
    x, y = np.meshgrid(range(0, width), range(0, height))
    xyz_camera = np.zeros((height, width, 4), dtype=np.float32)
    xyz_camera[:, :, 0] = (np.float32(x) / width - K[2]) * depth / K[0]
    xyz_camera[:, :, 1] = (np.float32(y) / height - K[3]) * depth / K[1]
    xyz_camera[:, :, 2] = depth

    if homo:
        dim = 4
        xyz_camera[:, :, 3] = np.float32(depth > 0)
    else:
        dim = 3
        xyz_camera = xyz_camera[:, :, :3]

    if flat:
        xyz_camera = xyz_camera.reshape((-1, dim))

    return xyz_camera


def gen_depth_map(proj, height, width, get_id, ref_depth=None):
    depth = np.zeros((height, width), dtype=np.float32)
    pix_num = proj.shape[1]
    index = np.zeros((height, width), dtype=np.int32)
    index_bool = np.zeros(pix_num, dtype=bool)
    threshold = 10

    for i in range(pix_num):
        x = int(proj[0, i])
        y = int(proj[1, i])
        if 0 < x and x <= width and 0 < y and y <= height:
            cur_depth = depth[y-1][x-1]
            if cur_depth == 0 or cur_depth > proj[2, i]:
                depth[y-1][x-1] = proj[2, i]
                index[y-1][x-1] = i

            if ref_depth is not None:
                if np.abs(proj[2, i] - ref_depth[y-1][x-1]) < threshold:
                    index_bool[i] = True

    if get_id == 1:
        return depth, index
    elif get_id == 2:
        return depth, index, index_bool
    else:
        return depth


def xyz2depth(points, intrinsic, shape, get_image_coor=False):
    """points is N x 3 numpy matrix
       intrinsic is the camera intrinsic

    """
    K = np.zeros((3, 3), dtype=np.float32)
    K[0, 0] = intrinsic[0]
    K[1, 1] = intrinsic[1]
    K[0, 2] = intrinsic[2]
    K[1, 2] = intrinsic[3]
    K[2, 2] = 1
    depth = np.zeros(shape, dtype=np.float32)
    project = np.dot(K, points.transpose())

    project[0:2, :] /= np.maximum(project[2, :], 1e-6)
    project[1, :] *= shape[0]
    project[0, :] *= shape[1]

    # pdb.set_trace()
    # depth, id_map = cut.gen_depth_map(project.astype(np.float32),
    #   shape[0], shape[1], 1)
    depth, id_map = gen_depth_map(project, shape[0], shape[1], 1)

    if get_image_coor:
        idx = np.array(range(0, shape[0] * shape[1]))
        idx = idx[depth.flatten() > 0]
        y = idx / shape[1]
        x = idx % shape[1]
        return depth, y, x
    else:
        return depth


def transform_c2w(xyz_camera, extr):
    # k x 3
    # extrinsic is world to camera transform
    xyz_world = np.dot(xyz_camera, extr[:, 0:3].transpose()) + \
             extr[:, 3].transpose()
    return xyz_world


def transform_w2c(xyz_world, extr):
    # k x 3
    # extrinsic is world to camera transform
    xyz_proj = np.dot((xyz_world - extr[:, 3]),
                       extr[:, 0:3])
    return xyz_proj


def depth2normal(depth, intrinsic):
    """ intrinsic must be normalized  fx / width, fy / height, cx / with
        cy / height
    """
    assert intrinsic[2] < 1 and intrinsic[3] < 1

    h, w = depth.shape
    intrinsic[[0, 2]] *= np.float32(w)
    intrinsic[[1, 3]] *= np.float32(h)
    normal = cut.depth2normals_np(depth, np.float32(intrinsic))
    normal = normal.transpose([1, 2, 0])

    return normal

def depth2flow(depth1, extr1, extr2, K, depth2=None, is_norm=False):
    """Convert depth from camera1 to camera 2 as flow for
       transforming the labels or warping images.

       extr1: either a 3x4 numpy matrix or a 6 dim numpy array with
              first 3 dim is rotation matrix and last 3 dim vector as
               translation (roll pitch yaw)
        K: is normalized intrinsic vector
        is_norm: whehter to normalize the flow with depth height and width
    """

    assert (K[2] <= 1. or K[3] <= 1.), "intrinsic must be normalized."

    if extr1.size == 6:
        # convert vector representation to matrix
        extr1_mat = np.zeros((3,4), dtype=np.float32)
        extr2_mat = np.zeros((3,4), dtype=np.float32)
        extr1_mat[:, :3] = euler_angles_to_rotation_matrix(extr1[:3])
        extr2_mat[:, :3] = euler_angles_to_rotation_matrix(extr2[:3])
        for i in range(3):
            extr1_mat[i, 3] = extr1[i + 3]
            extr2_mat[i, 3] = extr2[i + 3]
        extr1 = extr1_mat
        extr2 = extr2_mat

    K_mat = intrinsic_vec_to_mat(K, depth1.shape)
    height, width = depth1.shape
    pix_num = height * width

    xyz_camera = depth2xyz(depth1, K)
    valid = xyz_camera[:, 3] > 0
    xyz_camera = xyz_camera[valid, 0:3]
    xyz_world = transform_c2w(xyz_camera, extr1)
    xyz_camera2 = transform_w2c(xyz_world, extr2)

    project = np.dot(K_mat, xyz_camera2.transpose())
    project[0:2, :] /= project[2, :]

    x, y = np.meshgrid(range(1, width + 1), range(1, height + 1))
    x = x.reshape(pix_num)[valid].astype(np.float32)
    y = y.reshape(pix_num)[valid].astype(np.float32)

    # zbuffer here to prevent non-valid flow
    if depth2 is not None:
        proj_depth, _, index_bool = gen_depth_map(
            project.astype(np.float32), height, width, 2, depth2)
        mask = index_bool.reshape((height, width))
        index = np.flatnonzero(index_bool)
    else:
        proj_depth, index = cut.gen_depth_map(
                project.astype(np.float32), height, width, 1)
        index = index.flatten()
        index = index[index > 0]

    mask = np.zeros(pix_num)
    valid_index = np.int32((y[index] - 1) * width + x[index] - 1)
    mask[valid_index] = 1
    mask = mask.reshape((height, width))

    flowx = project[0, index] - x[index]
    flowy = project[1, index] - y[index]
    flow = np.zeros(2 * height * width, dtype=np.float32)

    valid = np.flatnonzero(valid)[index]
    flow[np.concatenate((valid, valid + height * width))] = \
            np.concatenate((flowx, flowy))
    flow = np.transpose(flow.reshape((2, height, width)), (1, 2, 0))

    if is_norm:
        flow[:, :, 0] /= np.float32(width)
        flow[:, :, 1] /= np.float32(height)

    return flow, proj_depth, mask


def warp2d(image, flow, is_forward=True):
    """Warp an image from source to target based on flow:

    Inputs:
        image: source image
        flow: flow from image to target x + flow is target

    Outputs:
        image_warp: warpped image
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    # print image.shape, flow.shape
    assert image.shape[:2] == flow.shape[:2]
    height, width = image.shape[0], image.shape[1]
    pix_num = height * width

    x, y = np.meshgrid(range(width), range(height))
    x = x.reshape(pix_num).astype(np.float32)
    y = y.reshape(pix_num).astype(np.float32)

    flow_new = flow.reshape((pix_num, -1))
    x_new = x + flow_new[:, 0]
    y_new = y + flow_new[:, 1]

    valid_x = np.logical_and(x_new >= 0, x_new < width)
    valid_y = np.logical_and(y_new >= 0, y_new < height)
    valid = np.logical_and(valid_x, valid_y)

    image_warp = np.zeros(image.shape, dtype=image.dtype)

    if is_forward:
        image_warp[np.int32(y_new[valid]), np.int32(x_new[valid]), ...] = \
            image[np.int32(y[valid]), np.int32(x[valid]), :]
    else:
        image_warp[np.int32(y[valid]), np.int32(x[valid]), ...] = \
            image[np.int32(y_new[valid]), np.int32(x_new[valid]), :]

    return np.squeeze(image_warp)


def grid_mesh(height, width, refer=None, threshold=None):
    """Generate triangle mesh for a grid layout
    """
    if not (refer is None):
      assert np.ndim(refer) == 2
      refer = refer.flatten()

    idx = np.array(range(0, height*width), dtype=np.int32)
    x = idx % width
    y = idx / width
    nei = [[[-1, 0], [0, -1]], [[0, 1], [1, 0]]]
    tri = []
    nx = np.newaxis
    for pts in nei:
      x_nei_1 = x + pts[0][0]
      y_nei_1 = y + pts[0][1]
      x_nei_2 = x + pts[1][0]
      y_nei_2 = y + pts[1][1]

      valid_s = np.logical_and(np.logical_and(x_nei_1 >= 0, y_nei_1 >= 0),
                               np.logical_and(x_nei_2 >= 0, y_nei_2 >= 0))
      valid_e = np.logical_and(np.logical_and(x_nei_1 < width, y_nei_1 < height),
                               np.logical_and(x_nei_2 < width, y_nei_2 < height))
      valid = np.logical_and(valid_s, valid_e)

      if not (refer is None):
        idx_tmp = idx[valid]
        idx_nei_1_tmp = y_nei_1[valid] * width + x_nei_1[valid]
        idx_nei_2_tmp = y_nei_2[valid] * width + x_nei_2[valid]

        max_diff = np.abs(refer[idx_tmp] - refer[idx_nei_1_tmp])
        max_diff = np.maximum(max_diff,
          np.abs(refer[idx_nei_1_tmp] - refer[idx_nei_2_tmp]))
        max_diff = np.maximum(max_diff,
          np.abs(refer[idx_tmp] - refer[idx_nei_2_tmp]))
        valid_2 = max_diff < threshold

        idx_tmp = idx_tmp[valid_2]
        valid_val = np.zeros(height*width, dtype=bool)
        valid_val[idx_tmp] = True
        valid = np.logical_and(valid, valid_val)

      cur = np.concatenate([idx[valid, nx],
          y_nei_1[valid, nx] * width + x_nei_1[valid, nx],
          y_nei_2[valid, nx] * width + x_nei_2[valid, nx]],
                            axis=1)
      tri.append(cur)

    return np.concatenate(tri, axis=0)


def random_perturb(trans, rot,
                   trans_perturb_range=5.,
                   rot_perturb_range=10.*np.pi/180,
                   noise_type='uniform'):
    """uniformly gittering to simulate GPS noise.
       need to copy input for generating the perturb
       Inputs:
           trans: np.array 3
           rot: np.array 3
       Outputs:
           perturbed trans and rot
    """

    if noise_type == 'uniform':
        sample = np.random.random_sample(2)
        angle = (sample[0] - 0.5) * np.pi
        dis =  sample[1] * trans_perturb_range
        trans_off = np.array([np.cos(angle) * dis, np.sin(angle) * dis])
        rot_off = (np.random.random_sample(2) - 0.5) * 2 * rot_perturb_range
    else:
        raise ValueError('No such noise type')

    trans_perturb = np.zeros(3)
    try:
        trans_perturb[:2] = trans[:2] + trans_off
    except:
        pdb.set_trace()
    trans_perturb[2] = trans[2]
    rot_perturb = rot.copy()
    rot_perturb[[0, 2]] = np.minimum(np.maximum(rot[[0, 2]] + rot_off, -1 * np.pi), np.pi)

    return trans_perturb, rot_perturb


def trans_vec_to_mat(rot, trans, dim=4):
    """ project vetices based on extrinsic parameters
    """
    mat = euler_angles_to_rotation_matrix(rot)
    mat = np.hstack([mat, trans.reshape((3, 1))])
    if dim == 4:
        mat = np.vstack([mat, np.array([0, 0, 0, 1])])

    return mat


def project(pose, scale, vertices):
    """ re posit and re scale the vertices of a 3D model
        based on extrinsic parameters
        pose: 0-3 rotation, 4-6 translation
    """

    if np.ndim(pose) == 1:
        mat = trans_vec_to_mat(pose[:3], pose[3:])
    else:
        mat = pose

    vertices = vertices * scale
    p_num = vertices.shape[0]

    # use this line because the model saving property
    points = vertices.copy()
    points = np.hstack([points, np.ones((p_num, 1))])

    # pdb.set_trace()
    points = np.matmul(points, mat.transpose())

    return points[:, :3]


def draw_3dboxes(image, boxes_3d, intrinsic, color=None):
    """ draw 3d boxes on image given intrinsic
        x1, y1, z1,  x2, y2, z2
    """
    def draw_quad(im, box_2d, color, lw):
        cv2.line(im, tuple(box_2d[:, 0]), tuple(box_2d[:, 1]), tuple(color), lw)
        cv2.line(im, tuple(box_2d[:, 0]), tuple(box_2d[:, 2]), tuple(color), lw)
        cv2.line(im, tuple(box_2d[:, 2]), tuple(box_2d[:, 3]), tuple(color), lw)
        cv2.line(im, tuple(box_2d[:, 1]), tuple(box_2d[:, 3]), tuple(color), lw)

    img = image.copy()
    color = [255, 0, 0] if color is None else color
    lw = 4
    for box_3d in boxes_3d:
        points = np.zeros((8,3))
        points[:, 0] = np.tile(box_3d[:, 0], 4)
        points[:, 1] = np.tile(np.repeat(box_3d[:, 1], 2), 2)
        points[:, 2] = np.repeat(box_3d[:, 2], 4)

        box_2d = intrinsic.dot(points.transpose())
        box_2d = np.maximum(box_2d / box_2d[2, :], 0)
        box_2d = np.uint32(box_2d[:2, :])

        draw_quad(img, box_2d[:, :4], color, lw)
        draw_quad(img, box_2d[:, 4:], color, lw)
        cv2.line(img, tuple(box_2d[:, 0]), tuple(box_2d[:, 4]), tuple(color), lw)
        cv2.line(img, tuple(box_2d[:, 2]), tuple(box_2d[:, 6]), tuple(color), lw)
        cv2.line(img, tuple(box_2d[:, 1]), tuple(box_2d[:, 5]), tuple(color), lw)
        cv2.line(img, tuple(box_2d[:, 3]), tuple(box_2d[:, 7]), tuple(color), lw)

    return img


def render_obj(pose, model, intrinsic, image_size, renderer):
    if not ('scale' in model):
        scale = [1.0, 1.0, 1.0]
    else:
        scale = model['scale']

    vert = project(pose, scale, model['vertices'])
    intrinsic = np.float64(
            intrinsic_mat_to_vec(intrinsic))
    depth, mask = renderer.renderMesh_np(
            np.float64(vert),
            np.float64(model['faces']),
            intrinsic, image_size[0], image_size[1])

    return depth, mask

if __name__ == '__main__':
    refer = np.zeros((5, 5))
    refer[3, 3] = 1
    threshold = 0.4
    print grid_mesh(5, 5, refer, threshold)

