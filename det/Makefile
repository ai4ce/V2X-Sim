# Path to the original V2X-Sim dataset
original_data_path := /scratch/dm4524/data/V2X-Sim-2
# Where to save the created data
create_data_save_path := /scratch/dm4524/data/V2X-Sim-det
scene_begin := 0 # Index of the begining scene
scene_end := 80 # Index of the ending scene + 1
from_agent := 0 # Index of the start agent
to_agent := 6 # Index of the end agent + 1

training_script := train_codet.py
training_data := $(create_data_save_path)/train # Path to the created training data

testing_script := test_codet.py
testing_data := $(create_data_save_path)/test # Path to the test/val data

bound := lowerbound # [lowerbound / upperbound]
com := v2v # [v2v / disco / when2com]
batch_size := 4
split := train # [train / test / val]

log_base := logs # Where to store the logs
epoch_num := 100 # Train for how many epochs
auto_resume_path := $(log_base) # If given, the model will resume from its most recent (by modification time) check point

create_data:
	python create_data_det.py --root $(original_data_path) --split $(split) --scene_begin $(scene_begin) --scene_end $(scene_end) --savepath $(create_data_save_path) --from_agent $(from_agent) --to_agent $(to_agent)

train_bound:
	python $(training_script) --data $(training_data) --bound $(bound) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num)

train_bound_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --log --batch $(batch_size) --no_cross_road --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num)

train:
	python $(training_script) --data $(training_data) --bound lowerbound --com $(com) --log --batch $(batch_size) --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num)

train_nc:
	python $(training_script) --data $(training_data) --bound lowerbound --com $(com) --log --batch $(batch_size) --no_cross_road --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num)

train_warp:
	python $(training_script) --data $(training_data) --bound $(bound) --com $(com) --log --batch $(batch_size) --warp_flag --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num)

train_warp_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --com $(com) --log --batch $(batch_size) --warp_flag --no_cross_road --auto_resume_path $(auto_resume_path) --nepoch $(epoch_num)

train_disco:
	python $(training_script) --data $(training_data) --bound $(bound) --com disco --log --batch $(batch_size) --kd_flag 1 --resume_teacher $(log_base)/upperbound/with_cross/epoch_$(epoch_num).pth --auto_resume_path $(auto_resume_path) --logpath $(log_base)  --nepoch $(epoch_num)

train_disco_nc:
	python $(training_script) --data $(training_data) --bound $(bound) --com disco --log --batch $(batch_size) --kd_flag 1 --resume_teacher	$(log_base)/upperbound/no_cross/epoch_$(epoch_num).pth --no_cross_road --auto_resume_path $(auto_resume_path) --logpath $(log_base)  --nepoch $(epoch_num)

test_bound:
	python $(testing_script) --data $(testing_data) --bound $(bound) --resume $(log_base)/$(bound)/with_cross/epoch_$(epoch_num).pth --tracking --logpath $(log_base)

test_bound_nc:
	python $(testing_script) --data $(testing_data) --bound $(bound) --resume $(log_base)/$(bound)/no_cross/epoch_$(epoch_num).pth --no_cross_road --tracking --logpath $(log_base)

test:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/$(com)/with_cross/epoch_$(epoch_num).pth --com $(com) --tracking --logpath $(log_base)

test_nc:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/$(com)/no_cross/epoch_$(epoch_num).pth --com $(com) --no_cross_road --tracking --logpath $(log_base)

test_w:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/when2com/with_cross/epoch_$(epoch_num).pth --com when2com --inference $(inference) --tracking --logpath $(log_base)

test_w_nc:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/when2com/no_cross/epoch_$(epoch_num).pth --com when2com --no_cross_road --inference $(inference) --tracking --logpath $(log_base)

test_warp:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/when2com_warp/with_cross/epoch_$(epoch_num).pth --com when2com --warp_flag --inference $(inference) --tracking --logpath $(log_base)

test_warp_nc:
	python $(testing_script) --data $(testing_data) --bound lowerbound --resume $(log_base)/when2com_warp/no_cross/epoch_$(epoch_num).pth --com when2com --warp_flag --no_cross_road --inference $(inference) --tracking --logpath $(log_base)
