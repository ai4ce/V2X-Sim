# V2X-Sim: A Virtual Collaborative Perception Dataset and Benchmark for Autonomous Driving

This repository provides a PyTorch benchmark implementation of the paper [V2X-Sim: A Virtual Collaborative Perception Dataset and Benchmark for Autonomous Driving](https://openreview.net/forum?id=15UnJrBjh_L)

<div align="center">
    <img src="https://ai4ce.github.io/V2X-Sim/img/multi-agent/overview.PNG" width="200" height="120"> 
    <img src="https://ai4ce.github.io/V2X-Sim/img/multi-agent/cars-1.PNG" width="200" height="120"> 
    <img src="https://ai4ce.github.io/V2X-Sim/img/multi-agent/infra-1.PNG" width="200" height="120">
</div>

## Abstract

Vehicle-to-everything (V2X), which denotes the collaboration via communication between a vehicle and any entity in its surrounding, can fundamentally improve the perception in self-driving systems. As the single-agent perception rapidly advances, collaborative perception has made little progress due to the shortage of public V2X datasets. In this work, we present V2X-Sim, the first public large-scale collaborative perception dataset in autonomous driving. V2X-Sim provides: 1) well-synchronized recordings from roadside infrastructure and multiple vehicles at the intersection to enable collaborative perception, 2) multi-modality sensor streams to facilitate multi-modality perception, and 3) diverse well-annotated ground truth to support various downstream tasks including detection, tracking, and segmentation. We seek to inspire research on multi-agent multi-modality multi-task perception, and our virtual dataset is promising to promote the development of collaborative perception before realistic datasets become widely available.



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

```

