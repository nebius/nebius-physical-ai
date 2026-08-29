"""Generated-script fragments for Isaac Lab trajectory export."""

TRAJECTORY_CAMERA_HELPERS = r'''
def _rollout_camera_view(task):
    if task == "Isaac-Cartpole-v0":
        # The manager-based Cartpole asset is rooted at z=2 m.  View it from
        # the side at the pole midpoint so cart translation and pole rotation
        # are both visible; the lower upstream RGB-camera pose points below
        # this task's elevated articulation and clips it at the top edge.
        return (0.0, -5.0, 3.0), (0.0, 0.0, 3.0)
    eye = (3.0, 3.0, 2.0)
    target = (0.0, 0.0, 0.8)
    return eye, target

def _normalize(vector):
    length = math.sqrt(sum(component * component for component in vector))
    if length < 1e-9:
        raise RuntimeError("cannot orient rollout camera from a zero-length vector")
    return [component / length for component in vector]

def _cross(left, right):
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]

def _look_at_quaternion(eye, target):
    # Isaac Lab's ``world`` camera convention looks along +X with +Z up.
    forward = _normalize([target[i] - eye[i] for i in range(3)])
    world_up = [0.0, 0.0, 1.0]
    dot_up = sum(world_up[i] * forward[i] for i in range(3))
    up = [world_up[i] - dot_up * forward[i] for i in range(3)]
    if math.sqrt(sum(component * component for component in up)) < 1e-6:
        up = [1.0, 0.0, 0.0]
        dot_up = sum(up[i] * forward[i] for i in range(3))
        up = [up[i] - dot_up * forward[i] for i in range(3)]
    up = _normalize(up)
    left = _cross(up, forward)
    matrix = [
        [forward[0], left[0], up[0]],
        [forward[1], left[1], up[1]],
        [forward[2], left[2], up[2]],
    ]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        )
    if matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return (
            (matrix[2][1] - matrix[1][2]) / scale,
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
        )
    if matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return (
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
        )
    scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return (
        (matrix[1][0] - matrix[0][1]) / scale,
        (matrix[0][2] + matrix[2][0]) / scale,
        (matrix[1][2] + matrix[2][1]) / scale,
        0.25 * scale,
    )

def _rgb_frame(render_env):
    camera = render_env.unwrapped.scene["npa_rollout_camera"]
    output = getattr(getattr(camera, "data", None), "output", None)
    if not output or "rgb" not in output:
        raise RuntimeError("Isaac rollout camera produced no RGB sensor output")
    frame = output["rgb"]
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise RuntimeError(f"Isaac rollout camera returned invalid shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame[..., :3])
'''.strip()
