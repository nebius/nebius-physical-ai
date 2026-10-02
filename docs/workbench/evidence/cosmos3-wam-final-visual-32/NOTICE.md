# Simulation and model sources

The robot scenes and task definitions come from the
[LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01),
revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`, distributed under its
[MIT license](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/LICENSE).
The images are new simulator renders made while executing predicted actions.

The policy uses [NVIDIA Cosmos Framework](https://github.com/NVIDIA/cosmos-framework/tree/cf5d68c00d97ccd2480a2320ed652b92dec63102),
Cosmos3-Nano weights and the LIBERO LeRobot v3 training dataset at the revisions
recorded in [evidence.json](evidence.json). NVIDIA distributes those sources
under [OpenMDW-1.1](../../../../skills/LICENSE-NVIDIA-COSMOS3-OPENMDW-1.1).
The predicted panels in the comparison videos are model-generated outputs.

Modifications: trained the policy on LIBERO-10, ran the pinned native evaluator,
converted its complete GIF recordings to H.264 MP4, and assembled recorded
frames into labeled contact sheets. No model weights or simulator asset files
are redistributed in this evidence directory.
