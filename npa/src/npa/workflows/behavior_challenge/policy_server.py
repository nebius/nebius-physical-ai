"""Match the pinned OpenPI observation names to the official R1Pro evaluator."""

from dataclasses import replace
import runpy


def _evaluator_robot(config):
    if config.name != "robot" or config.robot_type != "R1Pro":
        raise ValueError("Unexpected baseline robot configuration")
    links = {
        "image_0": "zed_link",
        "image_1": "left_realsense_link",
        "image_2": "right_realsense_link",
    }
    if set(config.observations) != set(links):
        raise ValueError("Unexpected baseline camera configuration")
    observations = {}
    for camera, link in links.items():
        observation = config.observations[camera]
        if observation.obs_key != f"robot::robot:{link}:Camera:0::rgb":
            raise ValueError("Unexpected baseline camera observation key")
        observations[camera] = replace(
            observation, obs_key=f"robot_r1::robot_r1:{link}:Camera:0::rgb"
        )
    return replace(config, name="robot_r1", observations=observations)


def _serve():
    from openpi.configs.robots import ROBOT_REGISTRY

    # Only policy lookup names change; the evaluator and observation values remain intact.
    ROBOT_REGISTRY["b1k/R1Pro"] = _evaluator_robot(ROBOT_REGISTRY["b1k/R1Pro"])
    runpy.run_path("scripts/b1k/serve_b1k.py", run_name="__main__")


if __name__ == "__main__":
    _serve()
