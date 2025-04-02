# @author Nico Schröder

import glob
import os
import sys

import torch
import torch.multiprocessing as mp
import numpy as np
from rich import print

from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.tracker import Tracker
from utils.tools import setup_experiment, split_chunks, load_decoders, remove_gpu_cache


# hardcode for now
result_folder = "experiments/hsfd_3103_2025-03-31_16-12-54"


def run():
    config = Config()
    yaml_files = glob.glob(f"{result_folder}/*.yaml")
    if len(yaml_files) > 1: # Check if there is exactly one YAML file
        sys.exit("There are multiple YAML files. Please handle accordingly.")
    elif len(yaml_files) == 0:  # If no YAML files are found
        sys.exit("No YAML files found in the specified path.")
    config.load(yaml_files[0])
    config.model_path = os.path.join(result_folder, "model", "pin_map.pth")

    print("[bold green]Load and inspect PIN Map[/bold green]","📍" )

    run_path = setup_experiment(config, sys.argv, debug_mode=True)

    mp.set_start_method("spawn") # don't forget this
    
    # initialize the mlp decoder
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    sem_mlp = Decoder(config, config.sem_mlp_hidden_dim, config.sem_mlp_level, config.sem_class_count + 1) if config.semantic_on else None
    color_mlp = Decoder(config, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel) if config.color_on else None
    
    mlp_dict = {}
    mlp_dict["sdf"] = geo_mlp
    mlp_dict["semantic"] = sem_mlp
    mlp_dict["color"] = color_mlp

    # initialize the neural point features
    neural_points: NeuralPoints = NeuralPoints(config)
    
    # load the map
    loaded_model = torch.load(config.model_path)
    neural_points = loaded_model["neural_points"]
    neural_points.travel_dist = {}
    neural_points.cur_ts = 0
    load_decoders(loaded_model, mlp_dict) 
    neural_points.temporal_local_map_on = False
    neural_points.recreate_hash(neural_points.neural_points[0], torch.eye(3).cuda(), False, False)
    neural_points.compute_feature_principle_components(down_rate = 59)
    tracker = Tracker(config, neural_points, {"sdf": geo_mlp, "semantic": None, "color": None})
    print("PIN Map loaded")
    
    @torch.no_grad()
    def query_sdf(points_np: np.ndarray):
        points = torch.tensor(points_np, dtype=torch.float32, device=config.device)
        sdf, *_ = tracker.query_source_points(
            coord=points,
            bs=64,  # batch size for inference
            query_sdf=True,
            query_sdf_grad=False,
            query_color=False,
            query_color_grad=False,
            query_sem=False,
            query_mask=False,
            query_certainty=False,
            query_locally=False
        )
        return sdf.cpu().numpy()
    
    # test SDF query
    test_points = np.random.uniform(
        low=[-5.0, -5.0, -0.5],
        high=[5.0, 5.0, 0.5],
        size=(100, 3)
    )
    sdf_values = query_sdf(test_points)
    for pt, sdf in zip(test_points, sdf_values):
        print(f"SDF at {pt} = {sdf:.3f} m")

    
if __name__ == "__main__":
    run()
    
    
