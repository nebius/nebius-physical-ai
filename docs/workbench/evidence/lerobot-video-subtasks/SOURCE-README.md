---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- tutorial
configs:
- config_name: default
  data_files: data/*/*.parquet
---

<!-- Local presentation change: the meta/info.json link below points to the
immutable upstream revision so it remains usable outside the source snapshot. -->

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](https://huggingface.co/datasets/lerobot/svla_so100_pickplace/blob/728583b5eaf9e739a7f119e2def466fa1d552402/meta/info.json):
```json
{
    "codebase_version": "v2.1",
    "robot_type": "so100",
    "total_episodes": 50,
    "total_frames": 19631,
    "total_tasks": 1,
    "total_videos": 100,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 30,
    "splits": {
        "train": "0:50"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "action": {
            "dtype": "float32",
            "shape": [
                6
            ],
            "names": [
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper"
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                6
            ],
            "names": [
                "main_shoulder_pan",
                "main_shoulder_lift",
                "main_elbow_flex",
                "main_wrist_flex",
                "main_wrist_roll",
                "main_gripper"
            ]
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "has_audio": false
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```
