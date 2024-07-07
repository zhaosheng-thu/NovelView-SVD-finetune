# SV3D finetuning and 1image2NV3D generation repository

## 1. finetuning for SV3D_u or SV3D_p published by stabilityai
Example: ![alt text](assets/image.png)

This case is fine-tuned on the Objaverse dataset rendered by <i>blender</i> from multi-views.

I changed the parameters in the config and added another condition: **height_z**. Since the initial 3D video can only be guided and controled by the azimuth and polar in version sv3d_p and sv3d_u, w/o the radius. Hence, I added it as the height_z and change the config files, to satisfy the `embed_size` of 1280.

## 2. NV3D pipeline

Coming soon.