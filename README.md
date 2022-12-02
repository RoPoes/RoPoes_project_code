# ROPOES: ROBOT JOINT POSE ESTIMATION

In this paper, we present a method to estimate the position state vector of the robotic manipulator. We propose a multi-camera setup to estimate joint angles and track joint poses in the scene. We use DRAKE for simulating KUKA IIWA and a two-camera system to track the manipulator as it moves in its configuration space. An hourglass network estimates 2D joint locations in a camera frame, and the intersection of their back-projected rays gives the 3D joint location estimates. We generate the dataset for training the hourglass network - 7K images of the manipulator in various configurations - by reorienting the camera to different locations over a hemispherical dome centered at the manipulator base. Ground truth(3D points) is generated from the simulator and trains the belief map estimation for each joint in the arm. We calculate the joint angles using the 3D key points and kinematic model. This work serves as the foundation for the egocentric supervision of robotic manipulators.

## Working

We train an hourglass network on our created dataset to predict the 2D keypoint locations in camera views for each joint in the manipulator. Currently, we place 2 cameras in the scene which capture the image of the manipulator and send it to the network which predict belief maps, which are locations of the joint keypoints in the image space. In challenging images, there could be multiple belief areas in these maps. Hence, these belief maps are further processed to get strongest peaks, pointing to the location of the joint keypoints. These are then passed to the triangulation module which estimates the 3D joint position of the manipulator. Using these 3D locations, joint angles are calculated between each link. Overall pipeline is summarized below:

![OverallPipeline](docs/ROPOES%20Pipeline.png)

## Results

Our proposed vision pipeline estimates joint states which includes 3D positions and angles and we are in process to extend it to real time. We propose dataset of 7k images of KUKA iiwa arm in drake simulator in egocentric way with different configurations of arm for handling difficult scenarios which DREAM\cite{c10} fails to capture. Our method returns the 3D joint positions with minor errors in orders of $10^{-2}$. Using joint positions, we calculate the joint angles between the links. Work done till present is aimed to use the joint states and angles for robot manipulation task of grasping and also estimate them in real time.

![Results-3D Positions and Angles](docs/ROPOES%203D%20Joints.png)