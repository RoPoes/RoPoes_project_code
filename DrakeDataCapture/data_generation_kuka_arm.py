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
sys.path.append('/home/jayaram/robot_manipulation_drake/ropoes_project_code/ropoes_snippets_temp/Literature-Review-Arm-Pose-Estimation/Ropoes')
from dream_code.scripts.network_inference import network_inference

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

class CameraSystem:
    def __init__(self, idx, meshcat, diagram, context):
        self.idx = idx
        
        # Read images
        depth_im_read = diagram.GetOutputPort("camera{}_depth_image".format(idx)).Eval(context).data.squeeze()
        self.depth_im = copy.deepcopy(depth_im_read)
        self.depth_im[self.depth_im == np.inf] = 10.0
        self.rgb_im = diagram.GetOutputPort('camera{}_rgb_image'.format(idx)).Eval(context).data

        # Visualize
        point_cloud = diagram.GetOutputPort("camera{}_point_cloud".format(idx)).Eval(context)
        meshcat.SetObject(f"Camera {idx} point cloud", point_cloud)

        # Get other info about the camera
        cam = diagram.GetSubsystemByName('camera' +str(idx))
        cam_context = cam.GetMyMutableContextFromRoot(context)
        self.X_WC = cam.body_pose_in_world_output_port().Eval(cam_context)
        self.cam_info = cam.depth_camera_info()
    
    def project_depth_to_pC(self, depth_pixel):
        """
        project depth pixels to points in camera frame
        using pinhole camera model
        Input:
            depth_pixels: numpy array of (nx3) or (3,)
        Output:
            pC: 3D point in camera frame, numpy array of (nx3)
        """
        # switch u,v due to python convention
        v = depth_pixel[:,0]
        u = depth_pixel[:,1]
        Z = depth_pixel[:,2]
        cx = self.cam_info.center_x()
        cy = self.cam_info.center_y()
        fx = self.cam_info.focal_x()
        fy = self.cam_info.focal_y()
        X = (u-cx) * Z/fx
        Y = (v-cy) * Z/fy
        pC = np.c_[X,Y,Z]
        return pC

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
def create_scene(sim_time_step=0.0001, n_cameras = 1, n_tracks = 1):
    # Clean up MeshCat.
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=sim_time_step)
    parser = Parser(plant)

    # Load iiwa arm in simulator.
    model_sdf = FindResourceOrThrow("drake/manipulation/models/iiwa_description/iiwa7/iiwa7_with_box_collision.sdf")
    iiwa_1 = parser.AddModelFromFile(model_sdf, model_name = "iiwa_1")

    # Weld the arm to the world so that it's fixed during the simulation.
    #model_frame = plant.GetFrameByName("table_top_center")
    #plant.WeldFrames(plant.world_frame(), model_frame)
    # Welding multi link robot on a particular pose
    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C=plant.GetFrameByName("iiwa_link_0", iiwa_1),
        X_PC=xyz_rpy_deg([0, 0, 0], [0, 0, 0]),
    )

    # Add Camera and renderer
    renderer_name = "renderer"
    scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
    world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())

    #create list of sensors
    sensors = []

    #generate points in a circle around the arm for a particular radius
    r = 2
    centerX = 0
    centerY = 0
    theta = np.linspace(0, 360, num=n_cameras)
    rad = np.linspace(2, 6, num = n_tracks)
    for r in rad:
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
            X_WB = xyz_rpy_deg([r, 0, 0.75], [-90, 0, 90])
            X_WB = delta_wrt_world_z  @  X_WB   

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
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"))

    diagram = builder.Build()

    # sensors = []
    # sensors.append(CameraSystem(0, meshcat, diagram, context))
    # sensors.append(CameraSystem(1, meshcat, diagram, context))

    return diagram, visualizer, scene_graph, sensors


def initialize_simulation(diagram):
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.)
    return simulator

def takePic(scene_graph, sensor, context, camera_number = 1, track_no = 1, sim_pic_count = 1):
    diagram_context = context #diagram.CreateDefaultContext()
    sensor_context = sensor.GetMyMutableContextFromRoot(diagram_context)
    sg_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)
    color = sensor.color_image_output_port().Eval(sensor_context).data
    depth = sensor.depth_image_32F_output_port().Eval(sensor_context).data.squeeze(2)
    label = sensor.label_image_output_port().Eval(sensor_context).data
    #save images
    cv2.imwrite("r_" + str(track_no) + "_c_" + str(camera_number) + "_" + "img_" + str(sim_pic_count) + ".jpg", cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA))
    # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # ax.imshow(color)
    # ax[1].imshow(depth)
    # ax[2].imshow(label)
    #plt.show()

    return color, depth

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

# Capture 
def generate_dataset(args, sim_time_step):
    images_sensors = []
    #initalize count of pics taken in simulation
    sim_count_pic = 1
    #transformations b/w every pair of cameras in multi camera system
    n_cameras = 10
    n_tracks = 4
    diagram, visualizer, scene_graph, sensors = create_scene(sim_time_step, n_cameras, n_tracks)

    print('total no of sensors: {}'.format(len(sensors)))
    simulator = initialize_simulation(diagram)
    for track_no in range(n_tracks):
        for camera_no in range(n_cameras):
            color, depth = takePic(scene_graph, sensors[(track_no * n_cameras) + camera_no], diagram.CreateDefaultContext(), camera_no, track_no, sim_count_pic)
            images_sensors.append((color, depth))

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
    visualizer.StartRecording()
    # simulator.AdvanceTo(0.250)
    # context = simulator.get_context()

    # color2, depth2 = takePic(scene_graph, sensors[i], context)

    # simulator.AdvanceTo(5.0)
    # context = simulator.get_context()
    # color3, depth3 = takePic(scene_graph, sensors[i], context)

    visualizer.PublishRecording()

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

