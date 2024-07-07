python scripts/distributed.py \
	--num_gpus 4 \
	--workers_per_gpu 4 \
	--num_images 21 \
	--blender_pth scripts/blender_script_.py \
	--input_models_path input_models_path.json \
	--valid_paths valid_paths.json \
	--output_dir ./views

python scripts/distributed.py \
	--num_gpus 10 \
	--workers_per_gpu 24 \
	--num_images 21 \
	--blender_pth scripts/blender_script_video.py \
	--input_models_path input_models_path.json \
	--valid_paths valid_paths_video_576.json \
	--output_dir ./views_video_576