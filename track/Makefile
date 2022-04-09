original_data_path := /scratch/dm4524/data/V2X-Sim-2 # Path to the original V2X-Sim dataset
det_data_path := /scratch/dm4524/data/V2X-Sim-det/ # Path to the dataset for detection

mode := disco # [lowerbound / upperbound/ v2v / disco / when2com / when2com_warp / who2com / who2com_warp]
cross := no_cross # [with_cross / no_cross]  with_cross ==> with RSU; no_cross ==> no RSU
current_agent := 0 # Index of current agent
split := test # [train / test / val]
scene_begin := 90 # Index of begining scene
scene_end := 100 # Index of ending scene

.PHONY: sort

create_data:
	python create_data_com.py --root $(original_data_path) --data $(det_data_path)/$(split) --split $(split) -b $(scene_begin) -e $(scene_end) --current_agent $(current_agent)

sort:
	cd sort && python sort.py --mode $(mode)/$(cross) --current_agent $(current_agent) --split $(split) -b $(scene_begin) -e $(scene_end)

# Evaludate MOTA, MOTP
eval1:
	cd TrackEval && python ./scripts/run_mot_challenge.py --BENCHMARK V2X --SPLIT_TO_EVAL $(split)$(current_agent) --TRACKERS_TO_EVAL sort-$(mode)/$(cross) --METRICS CLEAR --DO_PREPROC False

# Evaludate other matrics
eval2:
	cd TrackEval && python ./scripts/run_mot_challenge.py --BENCHMARK V2X --SPLIT_TO_EVAL $(split)$(current_agent) --TRACKERS_TO_EVAL sort-$(mode)/$(cross) --METRICS HOTA --DO_PREPROC False