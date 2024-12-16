import pin_slam
from eval.eval_traj_utils import get_metrics, mean_metrics
import torch
from IPython.display import display_markdown

print('Device used:', torch.cuda.get_device_name(0))

metrics_dicts = []
# seq_list = ['kaist01', 'kaist02', 'kaist03', 'dcc01', 'dcc02', 'dcc03', 'riverside01', 'riverside02', 'riverside03']
seq_list = ['riverside03']
for seq_str in seq_list:
    print('Now evaluate sequence '+ seq_str)
    seq_results = pin_slam.run_pin_slam('./config/lidar_slam/run_mulran.yaml', 'mulran', seq_str)
    metrics_dict = get_metrics(seq_results)
    metrics_dicts.append(metrics_dict)

metric_mean = mean_metrics(metrics_dicts)
table_results = f"# Experiment Results (MulRan dataset) \n|Metric|Value|\n|-:|:-|\n"
for metric, result in metric_mean.items():
    table_results += f"|{metric}|{result:.2f}|\n"
display_markdown(table_results, raw=True)