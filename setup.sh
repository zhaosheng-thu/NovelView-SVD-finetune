# install blender 3.2.2
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz
rm blender-3.2.2-linux-x64.tar.xz

# this is needed to download urls in blender
# https://github.com/python-poetry/poetry/issues/5117#issuecomment-1058747106
sudo update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs

sudo python3 scripts/start_xserver.py start || true
pip install -r requirements.txt

# download the models
# num_uids 798759
python scripts/download_objaverse.py --start_i 1 --end_i 2
python scripts/download_objaverse_hf.py --start_i 1 --end_i 10000

# blender
blender-3.2.2-linux-x64/blender -b -P scripts/blender_script_.py -- \
    --valid_json_path valid_paths.json \
    --output_dir ./views_new \
    --engine CYCLES \
    --scale 0.8 \
    --num_images 12 \
    --camera_dist 1.2 \
    --object_path downloaded_files/glbs/000-071/add3ff3c6ece459fb0ff7e66b9fe14f8.glb \
    --object_json_path input_models_path.json \

blender-3.2.2-linux-x64/blender -b -P scripts/blender_script_video.py -- \
    --valid_json_path valid_paths_video.json \
    --output_dir ./views_new_video \
    --engine CYCLES \
    --scale 0.8 \
    --num_images 21 \
    --camera_dist 1.2 \
    --object_path downloaded_files/glbs/000-071/add3ff3c6ece459fb0ff7e66b9fe14f8.glb \
    # --object_json_path input_models_path.json \