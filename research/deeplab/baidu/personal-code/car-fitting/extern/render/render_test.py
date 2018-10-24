import cv2
import numpy as np
import scipy.io as io
import render_egl
import data.kitti as kitti
import utils.utils as uts
import utils.utils_3d as uts_3d
import pdb


def project(p_in, T):
    # dimension of data and projection matrix
    dim_norm, dim_proj = T.shape

    p2_in = p_in.copy();
    v_num = p2_in.shape[0]
    if p2_in.shape[1] < dim_proj:
        p2_in = np.hstack([p2_in, np.ones([v_num, 1],
            dtype=np.float32)])

    p2_out = np.matmul(p2_in, T.transpose())
    p_out = p2_out[:, 0:-1] / p2_out[:,-1:];

    return p_out


def test_render():
    params = kitti.set_params_disp()
    car_model = ''
    car_model = params['car_model_path'] + '01.mat'
    model = io.loadmat(car_model)

    vertices = model['model']['hull'][0][0][0][0][1]
    faces = model['model']['hull'][0][0][0][0][0]
    scale = model['model']['scale'][0][0][0]
    # scale = np.float32([1,1,1])
    vertices = vertices[:, [0, 2, 1]]

    # pdb.set_trace()
    intrinsic = np.float64([250, 250, 160, 120]);
    imgsize = [240, 320];

    intrinsic = np.float64([350, 350, 320, 92]);
    imgsize = [180, 624];

    for i in range(100):
        ry = np.float(i)/10;
        # T = np.float32([0.0, ry, 0.0, 0.0, -0.3, 13.0])
        T = np.float32([0.0, 1.9, 0.0, -1.0, -1.1, 6.0])

        vertices_r = uts_3d.project(T, scale, vertices)
        # vertices_r[:, 1] = vertices_r[:, 1] - 0.3;
        # vertices_r[:, 2] = vertices_r[:, 2] + 3.0;

        faces = np.float64(faces)
        depth, mask = render_egl.renderMesh_np(
                vertices_r, faces, intrinsic, imgsize[0], imgsize[1])

        # uts.plot_images({'mask': mask,
        #                  'depth': depth})
        cv2.imwrite('test.png', depth)



if __name__ == '__main__':
    test_render()
