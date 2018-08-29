#
#  DeMoN - Depth Motion Network
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import pyximport; pyximport.install()
import numpy as np
import vtk

def angleaxis_to_rotation_matrix(aa):
    """Converts the 3 element angle axis representation to a 3x3 rotation matrix

    aa: numpy.ndarray with 1 dimension and 3 elements

    Returns a 3x3 numpy.ndarray
    """
    angle = np.sqrt(aa.dot(aa))

    if angle > 1e-6:
        c = np.cos(angle);
        s = np.sin(angle);
        u = np.array([aa[0]/angle, aa[1]/angle, aa[2]/angle]);

        R = np.empty((3,3))
        R[0,0] = c+u[0]*u[0]*(1-c);      R[0,1] = u[0]*u[1]*(1-c)-u[2]*s; R[0,2] = u[0]*u[2]*(1-c)+u[1]*s;
        R[1,0] = u[1]*u[0]*(1-c)+u[2]*s; R[1,1] = c+u[1]*u[1]*(1-c);      R[1,2] = u[1]*u[2]*(1-c)-u[0]*s;
        R[2,0] = u[2]*u[0]*(1-c)-u[1]*s; R[2,1] = u[2]*u[1]*(1-c)+u[0]*s; R[2,2] = c+u[2]*u[2]*(1-c);
    else:
        R = np.eye(3)
    return R


def compute_point_cloud_from_depthmap( depth, K, R, t, normals=None, colors=None ):
    """Creates a point cloud numpy array and optional normals and colors arrays

    depth: numpy.ndarray
        2d array with depth values

    K: numpy.ndarray
        3x3 matrix with internal camera parameters

    R: numpy.ndarray
        3x3 rotation matrix

    t: numpy.ndarray
        3d translation vector

    normals: numpy.ndarray
        optional array with normal vectors

    colors: numpy.ndarray
        optional RGB image with the same dimensions as the depth map.
        The shape is (3,h,w) with type uint8

    """
    from .vis_cython import compute_point_cloud_from_depthmap as _compute_point_cloud_from_depthmap
    return _compute_point_cloud_from_depthmap(depth, K, R, t, normals, colors)


def create_camera_actor(R, t):
    """Creates a vtkActor with a camera mesh"""
    cam_points = np.array([
        [0, 0, 0],
        [-1,-1, 1.5],
        [ 1,-1, 1.5],
        [ 1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0,1.2,1.5],
        [ 1,-0.5,1.5],
        [ 1, 0.5,1.5],
        [ 1.2, 0, 1.5]]
    )
    cam_points = (0.25*cam_points - t).dot(R)

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(cam_points.shape[0])
    for i in range(cam_points.shape[0]):
        vpoints.SetPoint(i, cam_points[i])
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    line_cells = vtk.vtkCellArray()

    line_cells.InsertNextCell( 5 );
    line_cells.InsertCellPoint( 1 );
    line_cells.InsertCellPoint( 2 );
    line_cells.InsertCellPoint( 3 );
    line_cells.InsertCellPoint( 4 );
    line_cells.InsertCellPoint( 1 );

    line_cells.InsertNextCell( 3 );
    line_cells.InsertCellPoint( 1 );
    line_cells.InsertCellPoint( 0 );
    line_cells.InsertCellPoint( 2 );

    line_cells.InsertNextCell( 3 );
    line_cells.InsertCellPoint( 3 );
    line_cells.InsertCellPoint( 0 );
    line_cells.InsertCellPoint( 4 );

    # x-axis indicator
    line_cells.InsertNextCell( 3 );
    line_cells.InsertCellPoint( 8 );
    line_cells.InsertCellPoint( 10 );
    line_cells.InsertCellPoint( 9 );

    # up vector (y-axis)
    poly_cells = vtk.vtkCellArray()
    poly_cells.InsertNextCell( 3 );
    poly_cells.InsertCellPoint( 5 );
    poly_cells.InsertCellPoint( 6 );
    poly_cells.InsertCellPoint( 7 );

    vpoly.SetLines(line_cells)
    vpoly.SetPolys(poly_cells)

    mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(vpoly)
    mapper.SetInput(vpoly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetLineWidth(2)

    return actor


def create_pointcloud_actor(points, colors=None):
    """Creates a vtkActor with the point cloud from numpy arrays

    points: numpy.ndarray
        pointcloud with shape (n,3)

    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkActor object
    """
    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    if not colors is None:
        vcolors = vtk.vtkUnsignedCharArray()
        vcolors.SetNumberOfComponents(3)
        vcolors.SetName("Colors")
        vcolors.SetNumberOfTuples(points.shape[0])
        for i in range(points.shape[0]):
            vcolors.SetTuple3(i ,colors[i,0],colors[i,1], colors[i,2])
        vpoly.GetPointData().SetScalars(vcolors)

    vcells = vtk.vtkCellArray()

    for i in range(points.shape[0]):
        vcells.InsertNextCell(1)
        vcells.InsertCellPoint(i)

    vpoly.SetVerts(vcells)

    mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(vpoly)
    mapper.SetInput(vpoly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(3)

    return actor


def visualize_depths(depths, intrinsics=None):
    """Visualizes the network predictions

       depths: dictionary of depth
       intrinsic: normalized intrinsic

       Compare multiple depth maps in with 3D point cloud
       Each depth map show with a given color
    """

    if intrinsics is None:
        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5])

    pointcloud = {'points': None, 'colors': None}
    R = np.eye(3)
    t = np.zeros((3,))

    for key, vis in depths.items():
        depth = vis['depth']
        h, w = vis['depth'].shape
        K = np.eye(3)
        K[0,0] = intrinsics[0]*w
        K[1,1] = intrinsics[1]*h
        K[0,2] = intrinsics[2]*w
        K[1,2] = intrinsics[3]*h

        img = np.ones([3, h, w])
        if 'color' in vis.keys():
            if vis['color'].ndim == 1:
                for i in [0, 1, 2]:
                    img[i, :, :] *= vis['color'][i]
            else:
                assert vis['color'].shape[0] == 3
                img = vis['color']
        else:
            img *= 255
        img = np.uint8(img)

        if pointcloud['points'] is None:
            pointcloud = compute_point_cloud_from_depthmap(
                    depth, K, R, t, None, img)
        else:
            pointcloud_cur = compute_point_cloud_from_depthmap(
                    depth, K, R, t, None, img)
            pointcloud['points'] = np.vstack([pointcloud_cur['points'],
                    pointcloud['points']])
            pointcloud['colors'] = np.vstack([pointcloud_cur['colors'],
                    pointcloud['colors']])


    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)

    pointcloud_actor = create_pointcloud_actor(
        points=pointcloud['points'],
        colors=pointcloud['colors'] if 'colors' in pointcloud else None)
    renderer.AddActor(pointcloud_actor)

    cam1_actor = create_camera_actor(R,t)
    renderer.AddActor(cam1_actor)

    axes = vtk.vtkAxesActor()
    axes.GetXAxisCaptionActor2D().SetHeight(0.05)
    axes.GetYAxisCaptionActor2D().SetHeight(0.05)
    axes.GetZAxisCaptionActor2D().SetHeight(0.05)
    axes.SetCylinderRadius(0.03)
    axes.SetShaftTypeToCylinder()
    renderer.AddActor(axes)

    renwin = vtk.vtkRenderWindow()
    renwin.SetWindowName("Point Cloud Viewer")
    renwin.SetSize(800,600)
    renwin.AddRenderer(renderer)


    # An interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interstyle = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interstyle)
    interactor.SetRenderWindow(renwin)

    # Start
    interactor.Initialize()
    interactor.Start()


def visualize_prediction(depth, intrinsics=None, normals=None,
        rotation=None, translation=None, image=None ):
    """Visualizes the network predictions

    inverse_depth: numpy.ndarray
        2d array with the inverse depth values with shape (h,w)

    intrinsics: numpy.ndarray
        4 element vector with the normalized intrinsic parameters with shape
        (4,)

    normals: numpy.ndarray
        normal map with shape (3,h,w)

    rotation: numpy.ndarray
        rotation in axis angle format with 3 elements with shape (3,)

    translation: numpy.ndarray
        translation vector with shape (3,)

    image: numpy.ndarray
        Image with shape (3,h,w) in the range [-0.5,0.5].
    """
    depth = depth.squeeze()

    w = depth.shape[-1]
    h = depth.shape[-2]

    if intrinsics is None:
        # sun3d intrinsics
        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5])

    K = np.eye(3)
    K[0,0] = intrinsics[0]*w
    K[1,1] = intrinsics[1]*h
    K[0,2] = intrinsics[2]*w
    K[1,2] = intrinsics[3]*h

    R1 = np.eye(3)
    t1 = np.zeros((3,))

    if not rotation is None and not translation is None:
        R2 = angleaxis_to_rotation_matrix(rotation.squeeze())
        t2 = translation.squeeze()
    else:
        R2 = np.eye(3)
        t2 = np.zeros((3,))

    if not normals is None:
        n = normals.squeeze()
    else:
        n = None

    if not image is None:
        img = ((image+0.5)*255).astype(np.uint8)
    else:
        img = None

    pointcloud = compute_point_cloud_from_depthmap(depth, K, R1, t1, n, img)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)

    pointcloud_actor = create_pointcloud_actor(
        points=pointcloud['points'],
        colors=pointcloud['colors'] if 'colors' in pointcloud else None,
        )
    renderer.AddActor(pointcloud_actor)

    cam1_actor = create_camera_actor(R1,t1)
    renderer.AddActor(cam1_actor)

    cam2_actor = create_camera_actor(R2,t2)
    renderer.AddActor(cam2_actor)

    axes = vtk.vtkAxesActor()
    axes.GetXAxisCaptionActor2D().SetHeight(0.05)
    axes.GetYAxisCaptionActor2D().SetHeight(0.05)
    axes.GetZAxisCaptionActor2D().SetHeight(0.05)
    axes.SetCylinderRadius(0.03)
    axes.SetShaftTypeToCylinder()
    renderer.AddActor(axes)

    renwin = vtk.vtkRenderWindow()
    renwin.SetWindowName("Point Cloud Viewer")
    renwin.SetSize(800,600)
    renwin.AddRenderer(renderer)


    # An interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interstyle = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(interstyle)
    interactor.SetRenderWindow(renwin)

    # Start
    interactor.Initialize()
    interactor.Start()


def transform_pointcloud_points(points, T):
    """Transforms the pointcloud with T

    points: numpy.ndarray
        pointcloud with shape (n,3)

    T: numpy.ndarray
        The 4x4 transformation

    Returns the transformed points
    """
    tmp = np.empty((points.shape[0],points.shape[1]+1),dtype=points.dtype)
    tmp[:,0:3] = points
    tmp[:,3] = 1
    return T.dot(tmp.transpose())[0:3].transpose()

