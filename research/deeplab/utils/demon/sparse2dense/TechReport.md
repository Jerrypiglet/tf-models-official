# Sparse to dense depth upsampling  depth image through videos


#### Abstract
Given a monocular or stereo video sequence, and a cheap sparse depthsensor, we want to produce high resolution and accurate depth results, which can be used in Augment Reality and Autonomous Driving for reducing cost of device.

Depth sensor in our system gives very sparse but accurate few 3D points, which provides guidance for scene scale and distortion.   While video have detailed and fine granularity information, using current deep learning strategy, we are be able to produce dense scene depth. However, in learned system, the video model is confusing about the absolute scale and distortion of the scene.  

Motivated by the complementary property between the two, we can combine the two methods by first train a learning framework, with a semi or full supervise way to produce depth.  Then we combine the sparse depth input.

Two methods:

- Joint training, input both video and sparse depth. (sparse depth only serve as a reference, need high resolution ground truth)

- Train only based on images, and warp to sparse depth. (Trust sparse depth more about the scale, no need high resolution ground truth)

####Related works
[Depth super-resolution] [1]
[Geometry warping][2] ( Laplacian mesh deform, ASAP deform )
[Floor plan to 3D][3]

#### Contribution:
In real application, we usually have sparse low resolution depth,
while high resolution images.

- depend on single image algorithm
- video might have better ability in predicting 3D structure.
- handle better in extreme sparse case than single image.


#### Solution
Formulation:
$\min_{\mathbf{X}}\sum_{i = 1}|\mathbf{x}_i - \mathbf{x}_{io}|  + \psi(\mathbf{X} | \mathbb{P})    $

$\mathbf{x}$ is a 3D point,  $\mathbf{x}_{io}$ is depth output from network and $\mathbb{P}$ is control points

By using as-rigid-as-possible, we have:
$E_\text{arap}(\mathbf{X}',\{\mathbf{R}_1,\dots,\mathbf{R}_{|T|}\}) = \sum\limits_{t \in T} a_t \sum\limits_{\{i,j\}
\in t} w_{ij} \left\|
\left(\mathbf{x}'_i - \mathbf{x}'_j\right)-
\mathbf{R}_t\left(\mathbf{x}_i - \mathbf{x}_j\right)\right\|^2.$

#### Experiments


#### Discuss



#### Reference
[1]:https://arxiv.org/abs/1607.01977  "Song et.al ECCV 2016"
[2]:http://www.jsoftware.us/vol7/jsw0709-21.pdf
[3]:http://www.cs.toronto.edu/~fidler/projects/rent3D.html
