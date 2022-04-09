# V2X-Sim: A Virtual Collaborative Perception Dataset and Benchmark for Autonomous Driving

This repository provides a PyTorch benchmark implementation of the paper [V2X-Sim: A Virtual Collaborative Perception Dataset and Benchmark for Autonomous Driving](https://openreview.net/forum?id=15UnJrBjh_L)

<div align="center">
    <img src="https://ai4ce.github.io/V2X-Sim/img/multi-agent/overview.PNG" width="200" height="120"> 
    <img src="https://ai4ce.github.io/V2X-Sim/img/multi-agent/cars-1.PNG" width="200" height="120"> 
    <img src="https://ai4ce.github.io/V2X-Sim/img/multi-agent/infra-1.PNG" width="200" height="120">
</div>

## Abstract

Vehicle-to-everything (V2X) communication techniques enable the collaboration between a vehicle and any other entity in its surrounding, which could fundamentally improve
the perception system for autonomous driving. However, the
lack of a public dataset significantly restricts the research
progress of collaborative perception. To fill this gap, we present
V2X-Sim, a comprehensive simulated multi-agent perception
dataset for V2X-aided autonomous driving. V2X-Sim provides:
(1) well-synchronized sensor recordings from road-side unit
(RSU) and multiple vehicles that enable multi-agent perception,
(2) multi-modality sensor streams that facilitate multi-modality
perception, and (3) diverse well-annotated ground truths that
support various perception tasks including detection, tracking,
and segmentation. Meanwhile, we build an open-source testbed
and provide a benchmark for the state-of-the-art collaborative
perception algorithms on three tasks, including detection, tracking and segmentation. V2X-Sim seeks to stimulate collaborative
perception research for autonomous driving before realistic
datasets become widely available.



## Dataset

You could find more detailed documents and the download link in our [website](https://ai4ce.github.io/V2X-Sim/index.html)!

<div align="center">
    <video loop autoplay muted>
        <source src="https://ai4ce.github.io/V2X-Sim/img/Media1.mp4" type="video/mp4">
    </video>
</div>



## Requirements

Tested with:

- Python 3.7
- PyTorch 1.8.0
- Torchvision 0.9.0
- CUDA 11.2



## Benchmark

We implement when2com, who2com, V2VNet, lowerbound and upperbound benchmark experiments on our datasets. You are welcome to go to [detection](./det), [segmentation](./seg) and [tracking](track) to find them.



## Acknowledgement

We are very grateful to multiple great opensourced codebases, without which this project would not have been possible:

- [NuSenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
- [sort](https://github.com/abewley/sort)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)



## Citation

If you find V2XSIM useful in your research, please cite:

```tex
@InProceedings{Li_2021_NeurIPS,
    title = {Learning Distilled Collaboration Graph for Multi-Agent Perception},
    author = {Li, Yiming and Ren, Shunli and Wu, Pengxiang and Chen, Siheng and Feng, Chen and Zhang, Wenjun},
    booktitle = {Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021)},
    year = {2021}
}
```

