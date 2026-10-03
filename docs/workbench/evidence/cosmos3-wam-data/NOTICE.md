# Dataset and transformation sources

The images and action values derive from NVIDIA Corporation's
[LIBERO LeRobot v3 dataset](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3/tree/e5907374380b8f96511957e6ba5582be52a1e179),
revision `e5907374380b8f96511957e6ba5582be52a1e179`, suite `libero_10`, episode 0.
The conversion derives from the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO).
The dataset is distributed under
[OpenMDW-1.1](../../../../skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1).

The native action conversion and normalization use
[NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework/tree/cf5d68c00d97ccd2480a2320ed652b92dec63102),
revision `cf5d68c00d97ccd2480a2320ed652b92dec63102`, also under OpenMDW-1.1.

Modifications: extracted two decoded camera frames and one sixteen-action
window, applied the native rotation and normalization transforms, and rendered
the values with explanatory labels. The images depict recorded demonstrations.
