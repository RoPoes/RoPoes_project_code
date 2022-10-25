# Literature-Review-Arm-Pose-Estimation
Literature Review and progress in estimation, keypoints/poses of articulated manipulators using cameras.

## Papers being/to-be reviewed

1. [DREAM: Deep Robot-to-Camera Extrinsics for Articulated Manipulators](https://github.com/NVlabs/DREAM) | [Paper](https://arxiv.org/pdf/1911.09231.pdf)  
  a. Detect Keypoints on MujoCo 3DOF manipulator through finetuning on DREAM.
2. [Dense Articulated Real-time Tracking](https://github.com/tschmidt23/dart) | [Paper](https://faculty.cc.gatech.edu/~afb/classes/CS7495-Fall2014/readings/dart.pdf)  
  a. Adapt DART's Depth based approach for pose tracking through multiple rgb cameras.
  b. DREAM depends on DART to generate ground truth.
  
  
## Possible Trajectory - To be refined after discussion.

1. DREAM paper does two things:
  i. Gets keypoints on the robotic arm  
  ii. Uses Forward Kinematics information to get 3D position of these keypoints and then apply PnP to get transformation
      from camera to robotic arm (to be confirmed if this transformation is from cam to robot's base or what) 
  
  Since, we are having two cameras:
    1. Finetuning DREAM should give us keypoints correctly in both views.  
    2. We can use MujoCo to give us FK in the same order to perform PnP  
    or  
    3. Experimental: Since we know the transformation of each camera with respect to each other, we may utilize this
       to skip FK usage. We have two views and corresponding keypoints in each view, we can generate the transformation
       from each camera to that keypoint.
       
  Problem still not addressed:  
    1. Orientation of the links still not found from above approach. (Ananth has an idea about this).  
    

