import torch


file_path = "/home/varaudio/Thang/test_something/Diffusion_Flow/image_fm_todo/results/cfg_fm-07-17-114436/last.ckpt"
dic = torch.load(file_path, map_location="cpu")
print(dic.keys())