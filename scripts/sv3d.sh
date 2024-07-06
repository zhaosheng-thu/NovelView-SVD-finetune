python scripts/sampling/simple_video_sample.py \
    --input_path /root/zyma/szhao-06/datasets/svd-sv3d/sv3d-0.jpg \
    --version sv3d_p \
    --elevations_deg [20,15,10,10,0,0,0,5,5,0,0,0,-5,-10,-10,-10,-15,-20,-20,-20,-20] \
    --azimuths_deg [1,20,35,50,60,75,90,110,125,140,160,180,200,210,220,240,260,280,300,320,340] \

python scripts/sampling/simple_video_sample.py \
    --input_path /root/zyma/szhao-06/datasets/svd-sv3d/sv3d-0.jpg \
    --version sv3d_u \

python tests/inference/test_sampling_video.py \
    --input_path /root/zyma/szhao-06/datasets/svd-sv3d/sv3d-0.jpg \
    --version sv3d_u \

python main.py \
    --name NV-SV3D \
    --base configs/example_training/svd/sv3d.yaml \
    --projectname SV3D-fine-tune \
    --no-test \
    --wandb

# with torch.no_grad():
#    return torch.stack(batch, 0, out=out)