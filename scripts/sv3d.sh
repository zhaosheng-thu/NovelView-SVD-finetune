python scripts/sampling/simple_video_sample.py \
    --input_path /root/zyma/szhao-06/datasets/svd-sv3d/sv3d-0.jpg \
    --version sv3d_p \
    --elevations_deg [0,15,10,10,0,0,0,5,5,0,0,0,-5,-10,-10,-10,-15,-20,-20,-20,-20] \
    --azimuths_deg [0,20,35,50,60,75,90,110,125,140,160,180,200,210,220,240,260,280,300,330,350] \

python scripts/sampling/simple_video_sample.py \
    --input_path /root/zyma/szhao-06/datasets/svd-sv3d/sv3d-0.jpg \
    --version sv3d_u \

python scripts/sampling/simple_video_sample.py \
    --input_path /root/zyma/szhao-06/datasets/svd-sv3d/sv3d-0.jpg \
    --version sv3d_p \
    --elevations_deg 1.0

python scripts/sampling/simple_3d_sample.py \
    --input_path /root/zyma/szhao-06/datasets/svd-sv3d/sv3d-0.jpg \
    --version sv3d_c \
    --elevations_deg 1.0

python main.py \
    --name SV3D-256 \
    --base configs/example_training/svd/sv3d.yaml \
    --projectname SV3D-fine-tune-256 \
    --no-test \
    --wandb

python main.py \
    --name SV3D-new-conditioner \
    --base configs/example_training/svd/sv3d-z.yaml \
    --projectname SV3D-conditioner-fine-tune \
    --no-test \
    --wandb

python main.py \
    --name NV \
    --base configs/example_training/svd/nv3d.yaml \
    --projectname NV3D \
    --no-test \
    --wandb
# with torch.no_grad():
#    return torch.stack(batch, 0, out=out)