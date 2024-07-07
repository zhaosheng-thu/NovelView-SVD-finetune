import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Process

def run_blender_script(blender_path, script_path, object_path, output_dir, engine, scale, num_images, camera_dist, gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    command = [
        blender_path, '-b', '-P', script_path, '--',
        '--object_path', object_path,
        '--output_dir', output_dir,
        '--engine', engine,
        '--scale', str(scale),
        '--num_images', str(num_images),
        '--camera_dist', str(camera_dist),
    ]
    subprocess.run(command, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_path", type=str, required=True, help="Path to the Blender executable")
    parser.add_argument("--script_path", type=str, required=True, help="Path to this script")
    parser.add_argument("--object_json_path", type=str, required=True, help="Path to the object JSON file")
    parser.add_argument("--output_dir", type=str, default="./views")
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--camera_dist", type=float, default=1.2)
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs available")
    
    args = parser.parse_args()
    
    with open(args.object_json_path) as f:
        object_paths = json.load(f)
    
    processes = []
    for i, object_path in enumerate(object_paths):
        gpu_id = i % args.num_gpus
        p = Process(target=run_blender_script, args=(
            args.blender_path, args.script_path, object_path, args.output_dir,
            args.engine, args.scale, args.num_images, args.camera_dist, gpu_id
        ))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()


"""
python multi_gpu_blender.py \
    --blender_path "blender-3.2.2-linux-x64/blender" \
    --script_path /path/to/your_script.py \
    --object_json_path input_model_paths.json \
    --output_dir ./views \
    --engine CYCLES \
    --scale 0.8 \
    --num_images 12 \
    --camera_dist 1.2 \
    --num_gpus 2

"""