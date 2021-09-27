## Tracking benchmark on V2XSIM

Here we implements the sort algorithm as our benchmark trackers and use the detection results obtained from [here](../det) to evaluate.

## Preparation
- Download V2XSIM datasets from our [website](https://ai4ce.github.io/V2X-Sim/index.html)
- Prepare tracking ground truth:
```bash
python create_data_com.py --root RAW_DATASET --data DETECTION_PREPROCESSED_DATASET --split test -b 80 -e 90
```

- Add nuscenes-devkit dependency: ```export PYTHONPATH=nuscenes-devkit/python-sdk/:PYTHONPATH```

## Evaluation

Run a tracker:
```bash
cd sort && python sort.py --mode MODE

# --mode [lowerbound/upperbound/when2com/who2com/when2com_warp/who2com_warp]
```



Evaluate tracking results:

```bash
cd TrackEval && python ./scripts/run_mot_challenge.py --BENCHMARK V2X --SPLIT_TO_EVAL test --TRACKERS_TO_EVAL YOUR_TRACKERS --METRICS CLEAR --DO_PREPROC False

# --TRACKERS_TO_EVAL [sort-lowerbound/sort-upperbound/sort-when2com/sort-who2com/sort-when2com_warp/sort-who2com_warp]
```



## Results

| Method      | MOTA  | MOTP  | HOTA  | DetA  | AssA  | DetRe | DetPr | AssRe | AssPr | LocA  |
| ----------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Lower-bound | 21.33 | 81.05 | 40.42 | 28.70 | 60.20 | 33.96 | 60.42 | 62.24 | 88.78 | 83.94 |
| When2com    | 35.3  | 81.18 | 44.40 | 32.87 | 63.31 | 34.80 | 77.60 | 64.65 | 93.21 | 83.80 |
| When2com*   | 35.97 | 81.20 | 44.97 | 33.02 | 64.83 | 34.72 | 78.63 | 66.24 | 93.13 | 83.73 |
| Who2com     | 34.27 | 81.08 | 44.07 | 32.47 | 63.02 | 34.34 | 78.07 | 64.32 | 93.41 | 84.18 |
| Who2com*    | 31.92 | 81.00 | 42.76 | 31.05 | 62.32 | 33.39 | 74.32 | 63.70 | 92.64 | 83.56 |
| V2VNet      | 35.07 | 80.96 | 44.50 | 32.93 | 63.76 | 34.84 | 77.18 | 65.17 | 93.02 | 83.45 |
| Upper-bound | 46.56 | 81.19 | 46.60 | 37.29 | 61.06 | 39.35 | 78.66 | 62.89 | 91.23 | 83.31 |

