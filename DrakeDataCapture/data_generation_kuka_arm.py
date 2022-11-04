# Import some basic libraries and functions for this tutorial.
from calendar import c
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import time
import argparse
import sys
import itertools
from copy import deepcopy
import cv2
import keyboard
import json
from IPython.display import clear_output, display
sys.path.append('/home/jayaram/robot_manipulation_drake/Ropoes_project_code/dream_code')
#from dream_code.scripts.network_inference import network_inference

from pydrake.common import FindResourceOrThrow, temp_directory
from pydrake.geometry import (
    MeshcatVisualizerCpp as MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
# from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.meshcat import JointSliders
# from pydrake.multibody.parsing import Parser
# from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
# from pydrake.systems.analysis import Simulator
# from pydrake.systems.framework import DiagramBuilder

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import MeshcatVisualizerCpp as MeshcatVisualizer, StartMeshcat
from pydrake.geometry.render import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    RenderCameraCore,
    RenderLabel,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdSensor,
)

from pydrake.all import InverseDynamicsController, LeafSystem, AbstractValue
from manipulation.scenarios import AddMultibodyTriad
#from manipulation.scenarios import AddRgbdSensors

# import torch
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from urllib.request import urlretrieve

# Create a Drake temporary directory to store files.
# Note: this tutorial will create two temporary files (cylinder.sdf and
# table_top.sdf) under `/tmp/robotlocomotion_drake_xxxxxx` directory.
temp_dir = temp_directory()

# Start the visualizer. The cell will output an HTTP link after the execution.
# Click the link and a MeshCat tab should appear in your browser.
meshcat = StartMeshcat()

# class CameraSystem:
#     def __init__(self, idx, meshcat, diagram, context):
#         self.idx = idx
        
#         # Read images
#         depth_im_read = diagram.GetOutputPort("camera{}_depth_image".format(idx)).Eval(context).data.squeeze()
#         self.depth_im = copy.deepcopy(depth_im_read)
#         self.depth_im[self.depth_im == np.inf] = 10.0
#         self.rgb_im = diagram.GetOutputPort('camera{}_rgb_image'.format(idx)).Eval(context).data

#         # Visualize
#         point_cloud = diagram.GetOutputPort("camera{}_point_cloud".format(idx)).Eval(context)
#         meshcat.SetObject(f"Camera {idx} point cloud", point_cloud)

#         # Get other info about the camera
#         cam = diagram.GetSubsystemByName('camera' +str(idx))
#         cam_context = cam.GetMyMutableContextFromRoot(context)
#         self.X_WC = cam.body_pose_in_world_output_port().Eval(cam_context)
#         self.cam_info = cam.depth_camera_info()
    
#     def project_depth_to_pC(self, depth_pixel):
#         """
#         project depth pixels to points in camera frame
#         using pinhole camera model
#         Input:
#             depth_pixels: numpy array of (nx3) or (3,)
#         Output:
#             pC: 3D point in camera frame, numpy array of (nx3)
#         """
#         # switch u,v due to python convention
#         v = depth_pixel[:,0]
#         u = depth_pixel[:,1]
#         Z = depth_pixel[:,2]
#         cx = self.cam_info.center_x()
#         cy = self.cam_info.center_y()
#         fx = self.cam_info.focal_x()
#         fy = self.cam_info.focal_y()
#         X = (u-cx) * Z/fx
#         Y = (v-cy) * Z/fy
#         pC = np.c_[X,Y,Z]
#         return pC

class PrintPose(LeafSystem):
    def __init__(self, body_index):
        LeafSystem.__init__(self)
        self._body_index = body_index
        self.DeclareAbstractInputPort("body_poses", AbstractValue.Make([RigidTransform()]))
        self.DeclareForcedPublishEvent(self.Publish)

    def Publish(self, context):
        pose = self.get_input_port().Eval(context)[self._body_index]
        clear_output(wait=True)
        print("gripper position (m): " + np.array2string(
            pose.translation(), formatter={
                'float': lambda x: "{:3.2f}".format(x)}))
        print("gripper roll-pitch-yaw (rad):" + np.array2string(
            RollPitchYaw(pose.rotation()).vector(),
                         formatter={'float': lambda x: "{:3.2f}".format(x)}))

# To inspect our own URDF files.
def model_inspector(filename): # filename will be model's sdf or urdf file
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    builder = DiagramBuilder() # Diagram is a directed graph of consituent systems where the output of the one system goes as input to another system.

    # Note: the time_step here is chosen arbitrarily.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    # Multiple geometry sources regsiter with the scene graph and they add pose and geometry like  velocity, position etc. through ports.
    # Plant and scene graph are also system and every system/diagram has its own context to represent its state.
    # Load the file into the plant/scene_graph.
    # Multi body plant contains multiple model instance.
    parser = Parser(plant) # 
    parser.AddModelFromFile(filename) # Model is added to the multi body plant. Parser creates a model instance from the sdf file.
    plant.Finalize()

    # Add two visualizers, one to publish the "visual" geometry, and one to
    # publish the "collision" geometry.
    visual = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"))
    collision = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kProximity, prefix="collision"))
    # Collision geometry is an approximation for the actual shape of the model.

    # ### Use a mesh as collision geometry
    # In some cases you need to have a detailed collision geometry for your simulation, e.g., in the case of dexterous manipulation for objects with an irregular shape, it might be justifiable to use a mesh as the collision geometry directly.
    # When an OBJ mesh is served as the collision geometry for a basic contact model, i.e., the point contact model, Drake will internally compute the convex hull of the mesh and use that instead. If you need a non-convex collision geometry, it's suggested to decompose your mesh to various convex shapes via a convex decomposition tool. There are many similar tools available that are mostly thin wrappers on [V-HACD](https://github.com/kmammou/v-hacd/). Among all, [convex_decomp_to_sdf](https://github.com/gizatt/convex_decomp_to_sdf) is a wrapper that we often use for Drake.
    # However, for a more complex contact model that Drake provides, such as the hydroelastic contact model, Drake can directly utilize the actual mesh for its contact force computation. Refer to [Hydroelastic user guide](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html) for more information.

    
    # Disable the collision geometry at the start; it can be enabled by the
    # checkbox in the meshcat controls.
    meshcat.SetProperty("collision", "visible", False)

    # Set the timeout to a small number in test mode. Otherwise, JointSliders
    # will run until "Stop JointSliders" button is clicked.
    default_interactive_timeout = 1.0 if "TEST_SRCDIR" in os.environ else None
    sliders = builder.AddSystem(JointSliders(meshcat, plant))
    diagram = builder.Build()
    sliders.Run(diagram, default_interactive_timeout)

def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def determine_R_t(R, t):
    #this returns R,t b/w 2 cameras (stereo calibration)
    transformations = {}
    assert(len(R) == len(t), 'no of rotation matrices should be same as no of translation vectors')
    # Rt = list(zip(R,t))
    for (camera_1_idx, camera_2_idx) in itertools.combinations(range(len(t)), 2):
        #we want to have diff combinations of R, t of cameras
        #first camera in pair
        R1 = R[camera_1_idx]
        t1 =t[camera_1_idx].reshape(3, 1)

        #second camera in pair
        R2 = R[camera_2_idx]
        t2 =t[camera_2_idx].reshape(3, 1)

        #compute R,t b/w 2 cameras in current pair
        T1 = np.vstack((np.hstack((R1, t1)), np.array([0, 0, 0, 1]).reshape(1, -1)))
        T2 = np.vstack((np.hstack((R2, t2)), np.array([0, 0, 0, 1]).reshape(1, -1)))
        T = np.linalg.inv(T2)@T1
        transformations[str(camera_1_idx) + '_' + str(camera_2_idx)] = (T[0:3, 0:3], T[:3, 3])

    return transformations

def determine_transformation_matrices(R, t):
    #returns transformation matrices for all cmaeras in sim setup as 3*4 matrices which is [R | t]
    transformations = []
    for i in range(len(R)):
        t[i] = t[i].reshape(-1, 1)
        transformations.append(np.hstack((R[i], t[i])))
    return transformations

# Returns a diagram which is consumed by simulator.
def create_scene(sim_time_step=0.0001, n_cameras = 1, rad = 1):
    # Clean up MeshCat.
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=sim_time_step)
    parser = Parser(plant)

    # Load iiwa arm in simulator.
    model_sdf = FindResourceOrThrow("drake/manipulation/models/iiwa_description/iiwa7/iiwa7_with_box_collision.sdf")
    model = parser.AddModelFromFile(model_sdf, model_name = "iiwa_1")   #model_instance

    # Weld the arm to the world so that it's fixed during the simulation.
    #model_frame = plant.GetFrameByName("table_top_center")
    #plant.WeldFrames(plant.world_frame(), model_frame)
    # Welding multi link robot on a particular pose
    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C=plant.GetFrameByName("iiwa_link_0", model),
        X_PC=xyz_rpy_deg([0, 0, 0], [0, 0, 0]),
    )

    # Add Camera and renderer
    renderer_name = "renderer"
    scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
    world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())

    #create list of sensors
    sensors = []
    #create list of projection matrices
    projection_matrices = []

    #generate points in a circle around the arm for a particular radius
    r = 0.25    #not required
    centerX = 0
    centerY = 0
    theta = np.linspace(0, 360, num=n_cameras)

    for t in theta:
        x = centerX + r * np.cos(t)
        y = centerY + r * np.sin(t)

        #camera intrinsics
        intrinsics = CameraInfo(  
            width=640,
            height=480,
            fov_y=np.pi/4,
        )
        # print(intrinsics_2)
        core = RenderCameraCore(
            renderer_name,
            intrinsics,
            ClippingRange(0.01, 10.0),
            RigidTransform(),
        )

        color_camera = ColorRenderCamera(core, show_window=False)
        depth_camera = DepthRenderCamera(core, DepthRange(0.01, 10.0))

        #fix the camera_2 position in world frame
        #X_WB = xyz_rpy_deg([2, 0, 0.75], [-90, 0, 90])   #for first camera
        # X_WB = xyz_rpy_deg([x, y, 0.75], [-90, 0, 90]) 
        delta_wrt_world_z = xyz_rpy_deg([0, 0, 0], [0, 0, t])   
        X_WB = xyz_rpy_deg([rad, 0, 0.75], [-90, 0, 90])
        X_WB = delta_wrt_world_z  @  X_WB  

        #determine P for this camera position
        R = np.array(X_WB.rotation().matrix())
        t = X_WB.translation()
        intrinsic_matrix = intrinsics.intrinsic_matrix()
        print('intrinsic matrix: {}'.format(intrinsic_matrix))
        P = intrinsic_matrix @ np.hstack((R.T, -R.T @ t.reshape(-1, 1)))
        projection_matrices.append(P)

        # print('rotation matrix: {}'.format(type(np.array(X_WB_2.rotation().matrix()))))
        #roll, pitch, yaw along X, Y Z directions (fixed angle representation)
        sensor = RgbdSensor(
            world_id,
            X_PB=X_WB,
            color_camera=color_camera,
            depth_camera=depth_camera,
        )
        builder.AddSystem(sensor)
        builder.Connect(
            scene_graph.get_query_output_port(),
            sensor.query_object_input_port(),
        )
        sensors.append(sensor)


    # Finalize the plant after loading the scene.
    plant.Finalize()
    # We use the default context to calculate the transformation of the table
    # in world frame but this is NOT the context the Diagram consumes.
    plant_context = plant.CreateDefaultContext()

    # Add visualizer to visualize the geometries.
    visualizer = None
    # visualizer = MeshcatVisualizer.AddToBuilder(
    #     builder, scene_graph, meshcat,
    #     MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"))
    
    #initialize controller for joints of model after plant is finalized with models
    iiwa_controller = iiwa_controller_fn(builder, plant, model)

    # Draw the frames
    # for body_name in ["iiwa_link_1", "iiwa_link_2", "iiwa_link_3", "iiwa_link_4", "iiwa_link_5", "iiwa_link_6", "iiwa_link_7"]:
    #     AddMultibodyTriad(plant.GetFrameByName(body_name), scene_graph)

    # gripper = plant.GetBodyByName("iiwa_link_3", model)
    #type of gripper: <class 'pydrake.multibody.tree.RigidBody_[float]'>

    # type:<class 'pydrake.multibody.tree.BodyFrame_[float]'  --> same instance type as plant.world_frame()
    # <BodyFrame_[float] name='iiwa_link_3' index=4 model_instance=2>
    # link_0_pose = iiwa_link_0_frame.CalcPoseInBodyFrame()

    # link_3_pose = iiwa_link_3_frame.GetFixedPoseInBodyFrame()
    # print('link_3_pose:{}'.format(link_3_pose))

    # print('gripper pose : {}'.format(gripper.get_pose_in_world()))
    # gripper_context = gripper.DoAllocateContext()
    # print(gripper)
    # gripper_context  = gripper.index()   #gripper.index() is LeafSystem
    # print_pose = builder.AddSystem(PrintPose(gripper.index()))
    # builder.Connect(plant.get_body_poses_output_port(),
    #                 print_pose.get_input_port())

    # PrintPose(gripper.index()).Publish(gripper_context)
    diagram = builder.Build()

    # sensors = []
    # sensors.append(CameraSystem(0, meshcat, diagram, context))
    # sensors.append(CameraSystem(1, meshcat, diagram, context))

    return diagram, builder, plant, plant_context, visualizer, scene_graph, iiwa_controller, sensors, model, projection_matrices


def initialize_simulation(diagram):
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.)
    return simulator

def takePic(scene_graph, sensor, context, camera_number = 1, track_no = 1, sim_pic_count = 1):
    diagram_context = context #diagram.CreateDefaultContext()
    sensor_context = sensor.GetMyMutableContextFromRoot(diagram_context)
    sg_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)
    color = sensor.color_image_output_port().Eval(sensor_context).data   #rgb image
    depth = sensor.depth_image_32F_output_port().Eval(sensor_context).data.squeeze(2)
    label = sensor.label_image_output_port().Eval(sensor_context).data
    #save images
    arm_conf_name = "r_" + str(track_no) + "_c_" + str(camera_number) + "_" + "img_" + str(sim_pic_count)
    cv2.imwrite(arm_conf_name + ".jpg", cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA))
    # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # ax.imshow(color)
    # ax[1].imshow(depth)
    # ax[2].imshow(label)
    #plt.show()

    return color, depth, arm_conf_name

def triangulation_DLT(P1, P2, kp1, kp2):
    A = [kp1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - kp1[0]*P1[2,:],
         kp2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - kp2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def get_3d_joints(plant_context, joint_frames):
    #compute relative T b.w each of joint of iiwa links wrt first joint of iiwa_link_0 as we welded first joint of iiwa_link_0 to world frame in plant
    joint_poses_wrt_joint0 = []
    for i in range(8):
        joint_poses_wrt_joint0.append((joint_frames[i].CalcPoseInWorld(plant_context)).translation())
    joint_poses_wrt_joint0 = np.array(joint_poses_wrt_joint0)
    return joint_poses_wrt_joint0

def iiwa_controller_fn(builder, plant, model):
    Kp = np.full(7, 100)
    Ki = 2 * np.sqrt(Kp)
    Kd = np.full(7, 1)
    iiwa_controller = builder.AddSystem(InverseDynamicsController(plant, Kp, Ki, Kd, False))
    iiwa_controller.set_name("iiwa_controller")
    #create feedback loop
    #plant state output to controller input
    #controller output to plant actuation input
    builder.Connect(plant.get_state_output_port(model),
                    iiwa_controller.get_input_port_estimated_state())
    builder.Connect(iiwa_controller.get_output_port_control(),
                    plant.get_actuation_input_port())

    return iiwa_controller

def iiwa_position_set(context, plant, diagram, iiwa_controller, position_vector, model):    
    #extract context from the diagram
    # context = diagram.CreateDefaultContext()
    #extract plant context from the full context
    plant_context = plant.GetMyMutableContextFromRoot(context)
    q0 = q0 = np.array(position_vector)
    x0 = np.hstack((q0, 0*q0))
    plant.SetPositions(plant_context, q0)

    #get joints/links info
    thetas = plant.GetPositions(plant_context)   #these will give angles
    print('thetas between links :{}'.format(thetas))    #this will print theta between diff links

    joint_pos = 1
    joint_indices = plant.GetJointIndices(model)
    joint = plant.get_joint(joint_indices[joint_pos])
    f = joint.frame_on_parent()
    # print('T matrix:{}'.format(f.GetFixedPoseInBodyFrame()))
    # print('joint positon :{}'.format(joint.GetOnePosition(plant_context)))  #this will give just single theta
    # print('joint at pos {}:{}'.format(joint_pos, joint))

    iiwa_controller.GetInputPort('desired_state').FixValue(iiwa_controller.GetMyMutableContextFromRoot(context), x0)
    return context, plant_context

def orient_arms(diagram, builder, plant, visualizer, scene_graph, iiwa_controller, model, sensor, context, plant_context, camera_no, track_no, sim_count_pic):
    #step1: get arm from scene
    diagram_context = context #diagram.CreateDefaultContext()
    sensor_context = sensor.GetMyMutableContextFromRoot(diagram_context)
    sg_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)
    # color = sensor.color_image_output_port().Eval(sensor_context).data   #rgb image
    # depth = sensor.depth_image_32F_output_port().Eval(sensor_context).data.squeeze(2)

    #step2: change individual link positions
    # iiwa_controller = iiwa_controller_fn(builder, plant, model)
    position_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # position_vector = [1, 0.7, 1, 1.3, 1.5, 0.65, 0.6]   #these are angles in radians at each joint
    context, plant_context = iiwa_position_set(context, plant, diagram, iiwa_controller, position_vector, model)

    #step3: get 3d points in that particular arm conf
    joint_frames = []  #these are frames at each of joint of model in plant
    for i in range(8):
        joint_frames.append(plant.GetFrameByName("iiwa_link_" + str(i) , model))
    key_points_3d = get_3d_joints(plant_context, joint_frames)
    # print('3d keypoints: \n {}'.format(key_points_3d))   #there are 8 joints in kuka arm

    #step4: Take pic after orienting diff inks in arm 
    # Note: latest context is required for takePic fn
    color, depth, arm_conf_name = takePic(scene_graph, sensor, context, camera_no, track_no, sim_count_pic)
    
    return key_points_3d, color, depth, arm_conf_name

def create_json(arm_conf_name, P, key_points_3d):  #n*3   (n = 7)
    n_points = key_points_3d.shape[0]
    kp_homogneous_3d = np.c_[key_points_3d, np.ones(n_points).reshape(-1, 1)]  #n*4
    # print('Projection matrix: {}'.format(P))
    kp_homogenous_2d = np.dot(P, kp_homogneous_3d.T)  #3*n
    kp_homogenous_2d = kp_homogenous_2d.T  #n*3
    kp_homogenous_2d = kp_homogenous_2d / (kp_homogenous_2d[: ,2].reshape(-1, 1))
    key_points_2d = kp_homogenous_2d[:, :2]  #n*2

    #overlay 2d kps on image (use below snippet just for testing)

    # current_img = cv2.imread(arm_conf_name + '.jpg')
    # print('img_shape:{}'.format(current_img.shape))
    # for kp_2d in key_points_2d:
    #     print('kp_2d: {}'.format(kp_2d))
    #     current_img = cv2.circle(current_img, (int(np.float32(kp_2d[0])), int(np.float32(kp_2d[1]))), radius=3, color=(0 ,0 ,255), thickness=-1)
    # # save with 2d points on image
    # cv2.imwrite(arm_conf_name + ".jpg", current_img)

    kps_list = []
    joint_names = ["iiwa7_link_0", "iiwa7_link_1", "iiwa7_link_2", "iiwa7_link_3", "iiwa7_link_4", "iiwa7_link_5", "iiwa7_link_6", "iiwa7_link_7"]
    for i,joint_name in enumerate(joint_names):
        kps_list.append({"name": joint_name,
					    "location": key_points_3d[i, :].tolist(),
					    "projected_location": key_points_2d[i, :].tolist()})

    #create json file in w mode and insert data 
    with open(arm_conf_name + '.json', 'w') as f:
        data_dict = {"camera_data":
                            {
                                "location_worldframe": [ -75.051300048828125, 47.982898712158203, 91.198799133300781 ],
                                "quaternion_xyzw_worldframe": [ 0.047899998724460602, 0.078100003302097321, -0.52090001106262207, 0.84869998693466187 ]
                            },
	                "objects": [
		                    {
                                "class": "kuka",
                                "keypoints": kps_list
                            }
                        ]
                   }
        json.dump(data_dict, f, indent = 5)

# Capture 
def generate_dataset(args, sim_time_step):
    images_sensors = []
    #initalize count of pics taken in simulation
    sim_count_pic = 1
    #transformations b/w every pair of cameras in multi camera system
    n_cameras = 15
    n_tracks = 3
    rad = np.linspace(3.86206897, 4.0, num = n_tracks)
    # [2.         2.06896552 2.13793103 2.20689655 2.27586207 2.34482759
    # 2.4137931  2.48275862 2.55172414 2.62068966 2.68965517 2.75862069
    # 2.82758621 2.89655172 2.96551724 3.03448276 3.10344828 3.17241379
    # 3.24137931 3.31034483 3.37931034 3.44827586 3.51724138 3.5862069
    # 3.65517241 3.72413793 3.79310345 3.86206897 3.93103448 4.        ]
    track_no = 27   #starting track no
    for r in rad:
        track_no = track_no + 1
        print('track no :{}'.format(track_no))
        diagram, builder, plant, plant_context, visualizer, scene_graph, iiwa_controller, sensors, model, projection_matrices = create_scene(sim_time_step, n_cameras, r)
        # simulator = initialize_simulation(diagram)

        for camera_no in range(n_cameras):
            n_times = 0
            n_arm_conf = 0  #conf of arm in current camera position
            for i in range(1):
                key_points_3d, color, depth, arm_conf_name = orient_arms(diagram, builder, plant, visualizer, scene_graph, iiwa_controller, model, sensors[camera_no], diagram.CreateDefaultContext(), plant_context, camera_no, track_no, sim_count_pic)
                images_sensors.append((color, depth))
                n_arm_conf = n_arm_conf + 1

                P = projection_matrices[camera_no]
                #create json file based on the above information of 3d key points of joints and Projection matrix of current sensor
                create_json(arm_conf_name, P, key_points_3d)

            # if(n_times == 1):
            #     break

            # # if(keyboard.is_pressed('q')):
            # #     break

            # n_times = n_times + 1

    # model_file = 'clutter_maskrcnn_model.pt'
    # if not os.path.exists(model_file):
    #     urlretrieve(
    #     "https://groups.csail.mit.edu/locomotion/clutter_maskrcnn_model.pt", model_file)
    # diagram, visualizer, scene_graph, sensors = create_scene(sim_time_step)

    # num_classes = 7
    # model = get_instance_segmentation_model(num_classes)
    # model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    # model.eval()

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model.to(device)
    # visualizer.StartRecording()
    # simulator.AdvanceTo(0.250)
    # context = simulator.get_context()

    # color2, depth2 = takePic(scene_graph, sensors[i], context)

    # simulator.AdvanceTo(5.0)
    # context = simulator.get_context()
    # color3, depth3 = takePic(scene_graph, sensors[i], context)

    # visualizer.PublishRecording()

    return images_sensors

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--input-config-path",
        default=None,
        help="Path to network configuration file. If nothing is specified, the script will search for a config file by the same name as the network parameters file.",
    )
    parser.add_argument(
        "-m", "--image_path", help="Path to image used for inference."
    )
    parser.add_argument(
        "-cam", "--camera_number", help="camera number"
    )
    parser.add_argument(
        "-sc", "--simulator_pic_count", help="simulator pic count."
    )
    parser.add_argument(
        "-k",
        "--keypoints_path",
        default=None,
        help="Path to NDDS dataset with ground truth keypoints information.",
    )
    parser.add_argument(
        "-g",
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="The GPU IDs on which to conduct network inference. Nothing specified means all GPUs will be utilized. Does not affect results, only how quickly the results are obtained.",
    )
    parser.add_argument(
        "-p",
        "--image-preproc-override",
        default=None,
        help="Overrides the image preprocessing specified by the network. (Debug argument.)",
    )
    args = parser.parse_args()

    # Run the simulation with a small time step. Try gradually increasing it!
    #capture images from diff sensors
    image_sensors = []   #this is list of image_sets taken from diff sensors. 
    image_sensors = generate_dataset(args, sim_time_step=0.0001)  



#python joints_extraction_3d.py -i ../dream_code/trained_models/kuka_dream_resnet_h.pth 

