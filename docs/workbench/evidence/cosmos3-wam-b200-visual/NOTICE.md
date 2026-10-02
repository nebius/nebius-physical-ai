# Visual sources

The source images and demonstration video are derived from NVIDIA's
[LIBERO LeRobot v3 dataset](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3/tree/e5907374380b8f96511957e6ba5582be52a1e179),
revision `e5907374380b8f96511957e6ba5582be52a1e179`, suite `libero_10`, episode 0.
NVIDIA Corporation is the dataset owner identified in its model card.
The conversion derives from the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO).
The dataset is distributed under [OpenMDW-1.1](../../../../skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1).

Modifications: extracted episode boundaries, transcoded AV1 to H.264, scaled
camera views, combined synchronized cameras, and added explanatory labels.
The complete episode contains 214 frames at 20 Hz. `input.png` is its first
front-camera frame. These are recorded demonstrations, not newly evaluated
policy rollouts.

Generated visuals use [NVIDIA Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano/tree/e59a53c25979a090fa8706c9acc0c254a6e89b92)
at revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`, with
[Cosmos Framework](https://github.com/NVIDIA/cosmos-framework/tree/cf5d68c00d97ccd2480a2320ed652b92dec63102)
at revision `cf5d68c00d97ccd2480a2320ed652b92dec63102`.
The base model and framework are distributed under OpenMDW-1.1.
