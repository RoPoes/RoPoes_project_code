# Import some basic libraries and functions for this tutorial.
import numpy as np
import os
import matplotlib.pyplot as plt

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

# Create a Drake temporary directory to store files.
# Note: this tutorial will create two temporary files (cylinder.sdf and
# table_top.sdf) under `/tmp/robotlocomotion_drake_xxxxxx` directory.
temp_dir = temp_directory()

# Start the visualizer. The cell will output an HTTP link after the execution.
# Click the link and a MeshCat tab should appear in your browser.
meshcat = StartMeshcat()


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



iiwa7_model_file = FindResourceOrThrow(
    "drake/manipulation/models/"
    "iiwa_description/iiwa7/iiwa7_with_box_collision.sdf")
#model_inspector(iiwa7_model_file)

# Create a simple cylinder model.
cylinder_sdf_file = os.path.join(temp_dir, "cylinder.sdf")
cylinder_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="cylinder">
    <pose>0 0 0 0 0 0</pose>
    <link name="cylinder_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.005833</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.005833</iyy>
          <iyz>0.0</iyz>
          <izz>0.005</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.2</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.2</length>
          </cylinder>
        </geometry>
        <material>
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>

"""

with open(cylinder_sdf_file, "w") as f:
    f.write(cylinder_sdf)

# Adding multiple objects to the scene
# Create a table top SDFormat model.
table_top_sdf_file = os.path.join(temp_dir, "table_top.sdf")
table_top_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="table_top">
    <link name="table_top_link">
      <visual name="visual">
        <pose>0 0 0.445 0 0 0</pose>
        <geometry>
          <box>
            <size>0.55 1.1 0.05</size>
          </box>
        </geometry>
        <material>
         <diffuse>0.9 0.8 0.7 1.0</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <pose>0 0 0.445  0 0 0</pose>
        <geometry>
          <box>
            <size>0.55 1.1 0.05</size>
          </box>
        </geometry>
      </collision>
    </link>
    <frame name="table_top_center">
      <pose relative_to="table_top_link">0 0 0.47 0 0 0</pose>
    </frame>
  </model>
</sdf>

"""

with open(table_top_sdf_file, "w") as f:
    f.write(table_top_sdf)


def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)


# Returns a diagram which is consumed by simulator.
def create_scene(sim_time_step=0.0001):
    # Clean up MeshCat.
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder, time_step=sim_time_step)
    parser = Parser(plant)

    # Loading models.
    # Load a cracker box from Drake. 
    # model_sdf = FindResourceOrThrow("drake/manipulation/models/iiwa_description/iiwa7/iiwa7_with_box_collision.sdf")
    # iiwa_1 = parser.AddModelFromFile(model_sdf, model_name = "iiwa_1")
    cracker_box = FindResourceOrThrow(
        "drake/manipulation/models/ycb/sdf/003_cracker_box.sdf")
    parser.AddModelFromFile(cracker_box)
    #Load the table top and the cylinder we created.
    parser.AddModelFromFile(cylinder_sdf_file)
    parser.AddModelFromFile(table_top_sdf_file)

    # Weld the table to the world so that it's fixed during the simulation.
    table_frame = plant.GetFrameByName("table_top_center")
    plant.WeldFrames(plant.world_frame(), table_frame)
    #Welding multi link robot on a particular pose
    # plant.WeldFrames(
    #   frame_on_parent_P=plant.world_frame(),
    #   frame_on_child_C=plant.GetFrameByName("iiwa_link_0", iiwa_1),
    #   X_PC=xyz_rpy_deg([0, -0.5, 0], [0, 0, 0]),
    # )

    # Add Camera and reenderer
    renderer_name = "renderer"
    scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

    intrinsics = CameraInfo(
    width=640,
    height=480,
    fov_y=np.pi/4,
    )
    core = RenderCameraCore(
        renderer_name,
        intrinsics,
        ClippingRange(0.01, 10.0),
        RigidTransform(),
    )
    color_camera = ColorRenderCamera(core, show_window=False)
    depth_camera = DepthRenderCamera(core, DepthRange(0.01, 10.0))

    # Adding camera to plant body, but it can also be added to a moving body.
    world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    X_WB = xyz_rpy_deg([2, 0, 0.75], [-90, 0, 90])
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


    # Finalize the plant after loading the scene.
    plant.Finalize()
    # We use the default context to calculate the transformation of the table
    # in world frame but this is NOT the context the Diagram consumes.
    plant_context = plant.CreateDefaultContext()

    # Set the initial pose for the free bodies, i.e., the custom box and the
    # cracker box.
    cylinder = plant.GetBodyByName("cylinder_link")
    X_WorldTable = table_frame.CalcPoseInWorld(plant_context)
    X_TableCylinder = RigidTransform(
        RollPitchYaw(np.asarray([90, 0, 0]) * np.pi / 180), p=[0,0,0.5])
    X_WorldCylinder = X_WorldTable.multiply(X_TableCylinder)
    plant.SetDefaultFreeBodyPose(cylinder, X_WorldCylinder) # Cylinder is a free body

    cracker_box = plant.GetBodyByName("base_link_cracker")
    X_TableCracker = RigidTransform(
        RollPitchYaw(np.asarray([45, 30, 0]) * np.pi / 180), p=[0,0,0.8])
    X_WorldCracker = X_WorldTable.multiply(X_TableCracker)
    plant.SetDefaultFreeBodyPose(cracker_box, X_WorldCracker)

    # Add visualizer to visualize the geometries.
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"))

    diagram = builder.Build()
    return diagram, visualizer, scene_graph, sensor


def initialize_simulation(diagram):
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.)
    return simulator

def takePic(scene_graph, sensor, context):
    diagram_context = context #diagram.CreateDefaultContext()
    sensor_context = sensor.GetMyMutableContextFromRoot(diagram_context)
    sg_context = scene_graph.GetMyMutableContextFromRoot(diagram_context)
    color = sensor.color_image_output_port().Eval(sensor_context).data
    depth = sensor.depth_image_32F_output_port().Eval(sensor_context).data.squeeze(2)
    label = sensor.label_image_output_port().Eval(sensor_context).data
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(color)
    ax[1].imshow(depth)
    ax[2].imshow(label)
    plt.show()
    return color, depth


# Capture 
def run_simulation(sim_time_step):
    diagram, visualizer, scene_graph, sensor = create_scene(sim_time_step)
    simulator = initialize_simulation(diagram)
    
    color1, depth1 = takePic(scene_graph, sensor, diagram.CreateDefaultContext())

    visualizer.StartRecording()
    simulator.AdvanceTo(0.250)
    context = simulator.get_context()

    color2, depth2 = takePic(scene_graph, sensor, context)

    simulator.AdvanceTo(5.0)
    context = simulator.get_context()
    color3, depth3 = takePic(scene_graph, sensor, context)

    visualizer.PublishRecording()

# Run the simulation with a small time step. Try gradually increasing it!
run_simulation(sim_time_step=0.0001)


