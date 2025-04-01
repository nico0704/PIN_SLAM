#### run ncd 128 ####
#### 1. pip3 install rosbags>=0.9.11
#### 2. rosbags-convert data/ncd_128/2021-07-01-10-37-38-quad-easy-001.bag --dst data/ncd_128/quad_e/ply
#### 3. python3 scripts/ros2bag2ply.py -i data/ncd_128/quad_e/quad_e.db3 -o data/ncd_128/quad_e/ply -t /os_cloud_node/points -p
#### 4. change in and out paths in dataset/converter/ncd128_pose_converter.py accordingly
#### 5. python3 dataset/converter/ncd128_pose_converter.py
#### 6. python3 run_ncd_128.py


import os
import torch
from IPython.display import display_markdown
import pin_slam
from eval.eval_traj_utils import get_metrics, mean_metrics

print('Device used:', torch.cuda.get_device_name(0))

metrics_dicts = []
# seq_list = ['quad_e', 'math_e', 'underground_e', 'cloister', 'stairs']
seq_list = ['quad_e']
# config_list = ['run_ncd_128', 'run_ncd_128', 'run_ncd_128', 'run_ncd_128_m', 'run_ncd_128_s']
config_list = ['run_ncd_128']
for (seq_str, config_str) in zip(seq_list, config_list):
    print('Now evaluate sequence '+ seq_str)
    seq_results = pin_slam.run_pin_slam(os.path.join('./config/lidar_slam', config_str + '.yaml'), 'ncd128', seq_str)
    metrics_dict = get_metrics(seq_results)
    metrics_dicts.append(metrics_dict)

metric_mean = mean_metrics(metrics_dicts)
table_results = f"# Experiment Results (Newer College 128 dataset) \n|Metric|Value|\n|-:|:-|\n"
for metric, result in metric_mean.items():
    table_results += f"|{metric}|{result:.2f}|\n"
display_markdown(table_results, raw=True)