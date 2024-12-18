import os
import torch
from IPython.display import display_markdown
import pin_slam
from eval.eval_traj_utils import get_metrics, mean_metrics

print('Device used:', torch.cuda.get_device_name(0))
print('Now evaluate sequence hsfd')
seq_results = pin_slam.run_pin_slam(os.path.join('./config/lidar_slam/run_hsfd.yaml'), 'hsfd')
