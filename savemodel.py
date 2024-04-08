# from avg_ckpts import ensemble

# class Args:
#     def __init__(self, exp_dir, exp_name, max_epochs, save_top_k):
#         self.exp_dir = exp_dir
#         self.exp_name = exp_name
#         self.trainer = self.Trainer(max_epochs)
#         self.checkpoint = self.Checkpoint(save_top_k)

#     class Trainer:
#         def __init__(self, max_epochs):
#             self.max_epochs = max_epochs

#     class Checkpoint:
#         def __init__(self, save_top_k):
#             self.save_top_k = save_top_k



# exp_dir = "/work/sunyiwei/12_10_work/CNVSRC2023Baseline/exp" 
# exp_name = "cncvs_4s" 
# max_epochs = 20 
# save_top_k = 5 

# args = Args(exp_dir, exp_name, max_epochs, save_top_k)

# model_path = ensemble(args)

# print(f"The averaged model is saved at {model_path}")
import torch
from avg_ckpts import average_checkpoints

paths = [
    # '/work/sunyiwei/12_10_work/CNVSRC2023Baseline/exp/cncvs_4s/epoch=16.ckpt',
    # '/work/sunyiwei/12_10_work/CNVSRC2023Baseline/exp/cncvs_4s/epoch=17.ckpt',
    # '/work/sunyiwei/12_10_work/CNVSRC2023Baseline/exp/cncvs_4s/epoch=18.ckpt',
    # '/work/sunyiwei/12_10_work/CNVSRC2023Baseline/exp/cncvs_4s/epoch=19.ckpt',
    # '/work/sunyiwei/12_10_work/CNVSRC2023Baseline/exp/cncvs_4s/epoch=20.ckpt',
    '/work/sunyiwei/12_10_work/CNVSRC2023Baseline/exp/cnvsrc-single/epoch=24.ckpt'
]

state_dicts = average_checkpoints(paths)
torch.save(state_dicts, './bitr_whole_epoch25.pth')