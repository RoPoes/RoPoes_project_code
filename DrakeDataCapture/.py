class SimSetup:
    def __init__(self, step, filename, base_link_frame_name, meshcat):
        self.step_size = step
        self.base_link_name = base_link_frame_name
        self.meshcat = meshcat
    def create_builder(self):
        #create a diagram builder object
        builder = DiagramBuilder()
        return builder
    def create_plant_scene_graph(self,builder):
        #plant and scene graph pointers for the newly created multibody plant and scene graph returned
        #they are added to the diagram
        #Makes a new MultibodyPlant with discrete update period time_step and adds it to a diagram builder together with the provided SceneGraph instance, connecting the geometry ports.
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=self.step_size)    
        return plant, scene_graph
    
    def create_model(self,plant, scene_graph, filename):
        model = Parser(plant, scene_graph).AddModelFromFile(FindResourceOrThrow(filename))
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(self.base_link_name))
        plant.Finalize()
        return model
    
    def add_meshcat2builder(self,builder, scene_graph, meshcat):
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
        return visualizer
    
    def setup(self):
        builder = self.create_builder()
        plant, scene_graph = self.create_plant_scene_graph(builder)
        model = self.create_model(plant, scene_graph,  filename)
        visualizer = self.add_meshcat2builder(builder, scene_graph,self.meshcat)
        
        return builder, plant, scene_graph, model, visualizer

# Capture 
def generate_dataset(args, sim_time_step):
    images_sensors = []
    #initalize count of pics taken in simulation
    sim_count_pic = 1
    #transformations b/w every pair of cameras in multi camera system
    n_cameras = 10
    n_tracks = 4
    diagram, visualizer, scene_graph, sensors, projection_matrices = create_scene(sim_time_step, n_cameras, n_tracks)

    print('total no of sensors: {}'.format(len(sensors)))
    simulator = initialize_simulation(diagram)
    for track_no in range(n_tracks):
        for camera_no in range(n_cameras):
            n_arm_conf = 0  #conf of arm in current camera position
            while(True):    # we can orient diff joints of arm and take pic (TakePic is called inside orient_arms)
                key_points_3d, color, depth, arm_conf_name = orient_arms(scene_graph, sensors[(track_no * n_cameras) + camera_no], diagram.CreateDefaultContext(), camera_no, track_no, sim_count_pic)
                images_sensors.append((color, depth))
                n_arm_conf = n_arm_conf + 1

                P = projection_matrices[(track_no * n_cameras) + camera_no]
                #create json file based on the above information of 3d key points of joints and Projection matrix of current sensor
                create_json(arm_conf_name, P, key_points_3d)

                if(keyboard.is_pressed('q')):
                    break

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

