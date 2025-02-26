# V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark for Autonomous Driving [RA-L 2022]

[Yiming Li](https://scholar.google.com/citations?user=i_aajNoAAAAJ), [Dekun Ma](https://dekun.me), [Ziyan An](https://ziyanan.github.io/), [Zixun Wang](), [Yiqi Zhong](https://www.linkedin.com/in/yiqi-zhong-078548129), [Siheng Chen](https://scholar.google.com/citations?user=W_Q33RMAAAAJ&hl=en), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ)

**"A comprehensive multi-agent multi-modal multi-task 3D perception dataset for autonomous driving."**

[![arXiv](https://img.shields.io/badge/Website-V2X--Sim-blue)](https://ai4ce.github.io/V2X-Sim/) 
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![GitLab issues total](https://badgen.net/github/issues/ai4ce/V2X-Sim)](https://gitlab.com/ai4ce/V2X-Sim/issues)
[![GitHub stars](https://img.shields.io/github/stars/ai4ce/V2X-Sim.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/ai4ce/V2X-Sim/stargazers/)
<div align="center">
    <img src="https://s2.loli.net/2022/06/15/cbs6hS2NHT7pDPL.png" height="300">
</div>
<br>

## News
**[2022-07]**  Our paper is available at [arxiv](https://arxiv.org/pdf/2202.08449.pdf).

**[2022-06]**  🔥 V2X-Sim is accepted at **IEEE Robotics and Automation Letters (RA-L)**.

## Abstract

Vehicle-to-everything (V2X) communication techniques enable the collaboration between a vehicle and any other
entity in its surrounding, which could fundamentally improve
the perception system for autonomous driving. However, the
lack of a public dataset significantly restricts the research
progress of collaborative perception. To fill this gap, we present
V2X-Sim, a comprehensive simulated multi-agent perception
dataset for V2X-aided autonomous driving. V2X-Sim provides:
(1) multi-agent sensor recordings from the road-side unit (RSU)
and multiple vehicles that enable collaborative perception, (2)
multi-modality sensor streams that facilitate multi-modality
perception, and (3) diverse ground truths that support various
perception tasks. Meanwhile, we build an open-source testbed
and provide a benchmark for the state-of-the-art collaborative
perception algorithms on three tasks, including detection, tracking and segmentation. V2X-Sim seeks to stimulate collaborative
perception research for autonomous driving before realistic
datasets become widely available.



## Dataset

Download links:
- Original dataset (you are going to parse this dataset yourself with `create_data.py` scripts for specific tasks): [Google Drive (US)](https://huggingface.co/datasets/ai4ce/V2X-Sim-2.0)  
- preprocessed datasets for detection and segmentation tasks and model checkpoints: [Google Drive (US)](https://drive.google.com/drive/folders/1NMag-yZSflhNw4y22i8CHTX5l8KDXnNd?usp=sharing)   

You could find more detailed documents on our [website](https://ai4ce.github.io/V2X-Sim/index.html)!

V2X-Sim follows the same file structure as the [Nuscenes dataset](https://www.nuscenes.org/):
```
V2X-Sim
├── maps # images for the map of one of the towns
├── sweeps # sensor data
|   ├── LIDAR_TOP_id_0 # top lidar data for the top camera, agent 0 (RSU)
|   ├── LIDAR_TOP_id_1 # top lidar data for the top camera, agent 1
|   ├── LIDAR_TOP_id_2 # top lidar data for the top camera, agent 2
|   ...
├── v1.0-mini # metadata
|   ├── scene.json # metadata for all the scenes
|   ├── sample.json # metadata for each sample, organized like linked-list
|   ├── sample_annotation.json # sample annotation metadata for each scene
|   ...
```

For parsed detection and segmentation dataset, the file structure will be:
```
V2X-Sim-det / V2X-Sim-seg
├── train # training data
|   ├── agent0 # data for RSU
|   |   ├── 0_0 # scene 0, frame 0
|   |   ├── 0_1 # scene 0, frame 1
|   |   |   ...
|   ├── agent1 # data for agent 1
|   ...
|   ├── agent5 # data for agent 5
├── val # validation data
├── test # test data
```


https://user-images.githubusercontent.com/53892579/180342204-1697f102-5f69-45d1-a62e-9460f4628fb8.mp4

https://user-images.githubusercontent.com/53892579/180342351-ef58e302-9bcb-47fa-a80f-1fe49ee80152.mp4

https://user-images.githubusercontent.com/53892579/180341986-1389ba9a-2bab-427f-8873-7cd7cba38fbe.mp4

## Requirements

Tested with:

- Python 3.7
- PyTorch 1.8.0
- Torchvision 0.9.0
- CUDA 11.2



## Benchmark

We implement when2com, who2com, V2VNet, lowerbound and upperbound benchmark experiments on our datasets. You are welcome to go to `README` files in [detection](https://github.com/coperception/coperception/tree/master/tools/det), [segmentation](https://github.com/coperception/coperception/tree/master/tools/seg) and [tracking](https://github.com/coperception/coperception/tree/master/tools/track) to find them.



## Acknowledgement

We are very grateful to multiple great opensourced codebases, without which this project would not have been possible:

- [NuSenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
- [sort](https://github.com/abewley/sort)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [coperception](https://github.com/coperception/coperception)

## Citation

If you find V2XSIM useful in your research, please cite:

```bibtex
@article{li2022v2x,
  title={V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark for Autonomous Driving},
  author={Li, Yiming and Ma, Dekun and An, Ziyan and Wang, Zixun and Zhong, Yiqi and Chen, Siheng and Feng, Chen},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={4},
  pages={10914--10921},
  year={2022},
  publisher={IEEE}
}
```

