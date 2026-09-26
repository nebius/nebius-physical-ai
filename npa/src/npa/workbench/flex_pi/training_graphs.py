"""Capture static mixed-attention training while retaining eager control and evaluation."""

from npa.workbench.flex_pi.runtime import FlexPiError


def validate_cuda_graphs(mode, activation_checkpointing):
    """Validate the opt-in training capture boundary before vendor execution.

    Args:
        mode: Off for eager execution or mot for the mixed-attention module.
        activation_checkpointing: Requested activation recomputation policy.
    Returns:
        None.
    Raises:
        FlexPiError: The policy or recomputation combination is unsupported.
    """
    if mode not in {"off", "mot"}:
        raise FlexPiError("cuda-graphs must be off or mot")
    if mode == "mot" and activation_checkpointing != "off":
        raise FlexPiError("cuda-graphs mot requires activation-checkpointing off")


def configure_training_graphs(model, configuration):
    """Install a training-only capture wrapper before distributed wrapping.

    Args:
        model: Instantiated Flex-Pi model before Accelerator.prepare().
        configuration: Resolved upstream training configuration.
    Returns:
        None; model parameters and state-dict keys are preserved.
    Raises:
        FlexPiError: Capture policy, vendor runtime or model state is unsupported.
    """
    mode = configuration.get("npa_cuda_graphs", "off")
    validate_cuda_graphs(mode, configuration.get("npa_activation_checkpointing", "on"))
    if mode == "off":
        return
    if getattr(model, "_npa_training_graphs", None) is not None:
        raise FlexPiError("training CUDA graphs were already configured")
    _check_runtime()
    state = _TrainingGraphs(model.mot)
    model.mot.forward = state.forward
    model._npa_training_graphs = state


def _check_runtime():
    import torch
    import torch._dynamo

    if torch.__version__.split("+")[0] != "2.7.1" or not torch.cuda.is_available():
        raise FlexPiError("training CUDA graphs require the pinned CUDA Torch 2.7.1")
    if torch._dynamo.config.suppress_errors:
        raise FlexPiError("training CUDA graphs forbid compiler error suppression")


def prepare_ddp_for_graph_capture(accelerator):
    """Construct native DDP on a side stream before its parameters enter capture.

    Args:
        accelerator: Pinned Accelerator whose initial prepare has not run yet.
    Returns:
        None; installs a one-use wrapper and restores prepare even on failure.
    Raises:
        RuntimeError: Stream setup or the original prepare operation fails.
    """
    import torch

    original = accelerator.prepare

    def prepare(*args, **kwargs):
        accelerator.prepare = original
        caller = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(caller)
        try:
            with torch.cuda.stream(stream):
                result = original(*args, **kwargs)
        finally:
            caller.wait_stream(stream)
        return result

    accelerator.prepare = prepare


def _input_signature(leaves):
    import torch

    signature = []
    for value in leaves:
        if isinstance(value, torch.Tensor):
            signature.append(
                (
                    "tensor",
                    value.shape,
                    value.stride(),
                    value.dtype,
                    value.device,
                    value.requires_grad,
                )
            )
        elif type(value) in {int, float, bool, str, type(None)}:
            signature.append((type(value).__name__, value))
        else:
            raise FlexPiError("CUDA graph input contains unsupported static metadata")
    return signature


def _native_graphs(wrapper):
    import inspect
    import torch

    # The closure layout is verified against the pinned Torch 2.7.1 source.
    # Count successful native replay calls, not merely an API invocation.
    try:
        function = inspect.getclosurevars(wrapper.forward).nonlocals["graphed"]
        autograd = inspect.getclosurevars(function).nonlocals["Graphed"]
        forward = inspect.getclosurevars(autograd.forward).nonlocals["fwd_graph"]
        backward = inspect.getclosurevars(inspect.unwrap(autograd.backward)).nonlocals[
            "bwd_graph"
        ]
    except (KeyError, TypeError, AttributeError) as error:
        raise FlexPiError("pinned native CUDA graph closure layout differs") from error
    if forward is backward or not all(
        isinstance(g, torch.cuda.CUDAGraph) for g in (forward, backward)
    ):
        raise FlexPiError("native capture lacks distinct forward/backward CUDA graphs")
    # Torch 2.7.1 defines Graphed inside make_graphed_autograd_function: this
    # class belongs to one capture, not torch.autograd or unrelated models.
    if getattr(autograd, "_npa_replays_observed", False):
        raise FlexPiError("native CUDA graph replays were already configured")
    return autograd, forward, backward


def _observe_native_replays(wrapper, counts):
    from functools import wraps

    autograd, forward, backward = _native_graphs(wrapper)
    for mode, graph in (("forward", forward), ("backward", backward)):
        replay = graph.replay

        def observed(*, replay=replay, mode=mode):
            replay()
            counts[mode] += 1

        graph.replay = observed

    native_backward = autograd.backward

    @wraps(native_backward)
    def independent_backward(*args):
        gradients = native_backward(*args)
        # Native backward returns detached graph-owned buffers. AccumulateGrad
        # may adopt those buffers as parameter.grad, so the next replay would
        # overwrite prior microbatches. Return independent storage instead.
        return tuple(grad.clone() if grad is not None else None for grad in gradients)

    autograd.backward = staticmethod(independent_backward)
    autograd._npa_replays_observed = True


def _capture_native(state, leaves, tree_spec):
    import torch
    from torch.utils._pytree import tree_unflatten

    positions = [i for i, value in enumerate(leaves) if isinstance(value, torch.Tensor)]
    static = tuple(
        leaves[i].detach().clone().requires_grad_(leaves[i].requires_grad)
        for i in positions
    )
    template = [None if isinstance(value, torch.Tensor) else value for value in leaves]
    forward_autocast = torch.is_autocast_enabled("cuda")
    forward_dtype = torch.get_autocast_dtype("cuda")

    class CaptureModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot = state.module

        def forward(self, *tensors):
            values = list(template)
            for index, tensor in zip(positions, tensors, strict=True):
                values[index] = tensor
            args, kwargs = tree_unflatten(values, tree_spec)
            with torch.autocast(
                "cuda",
                enabled=forward_autocast,
                dtype=forward_dtype,
                cache_enabled=False,
            ):
                return state.original(*args, **kwargs)

    wrapper = CaptureModule()
    # Warmup/capture must not consume the live training RNG sequence. Input
    # leaves are detached copies so qualification cannot retain an eager graph.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        # The native API also captures autograd.grad. Match eager training:
        # autocast applies only to forward, never to backward capture.
        with torch.autocast("cuda", enabled=False):
            wrapper = torch.cuda.make_graphed_callables(
                wrapper, static, num_warmup_iters=11, allow_unused_input=True
            )
    _observe_native_replays(wrapper, state.replays)
    return wrapper, positions


class _TrainingGraphs:
    def __init__(self, module):
        self.module = module
        self.original = module.forward
        self.training_calls = 0
        self.evaluation_calls = 0
        self.replays = {"forward": 0, "backward": 0}
        self.captured = None
        self.signature = None
        self.tree_spec = None
        self.positions = None

    def check(self, completed=False):
        forward, backward = self.replays["forward"], self.replays["backward"]
        if forward != self.training_calls or forward - backward not in (
            {0} if completed else {0, 1}
        ):
            raise FlexPiError("native CUDA graph replay or backward completion differs")

    def forward(self, *args, **kwargs):
        import torch
        from torch.utils._pytree import tree_flatten

        self.check(completed=True)
        # Evaluation keeps the exact eager path; only training needs capture.
        if not self.module.training or not torch.is_grad_enabled():
            self.evaluation_calls += 1
            return self.original(*args, **kwargs)
        leaves, spec = tree_flatten((args, kwargs))
        signature = _input_signature(leaves)
        if self.captured is None:
            self.captured, self.positions = _capture_native(self, leaves, spec)
            self.signature, self.tree_spec = signature, spec
        elif signature != self.signature or spec != self.tree_spec:
            raise FlexPiError(
                "training input differs from the static CUDA graph contract"
            )
        with torch.autocast("cuda", cache_enabled=False):
            result = self.captured(*(leaves[i] for i in self.positions))
        self.training_calls += 1
        self.check()
        return result

    def receipt(self):
        self.check(completed=True)
        if self.training_calls < 2 or self.captured is None:
            raise FlexPiError("training lacks replayed forward/backward CUDA graphs")
        return {
            "policy": "mot",
            "backend": "make_graphed_callables",
            "fullgraph": True,
            "dynamic": False,
            "evaluation": "eager",
            "captured_training_calls": self.training_calls,
            "eager_evaluation_calls": self.evaluation_calls,
            "native_replays": dict(self.replays),
            "recorded_cuda_graphs": {"forward": 1, "backward": 1, "inference": 0},
            "skipped_cuda_graphs": 0,
        }


def check_training_graphs(model):
    """Require matching native graph replay and backward completion.

    Args:
        model: Original model, or its native DDP wrapper.
    Returns:
        None; eager models require no graph inspection.
    Raises:
        FlexPiError: The requested native forward/backward replay did not occur.
    """
    unwrapped = getattr(model, "module", model)
    state = getattr(unwrapped, "_npa_training_graphs", None)
    if state is not None:
        state.check(completed=True)


def training_graphs_receipt(model, configuration):
    """Gather every rank's observed capture state before publishing a phase.

    Args:
        model: Original model, or its native DDP wrapper.
        configuration: Resolved training configuration.
    Returns:
        The requested policy and independently checked rank receipts.
    Raises:
        FlexPiError: Any rank lacks the requested policy or actual CUDA graphs.
    """
    import torch

    unwrapped = getattr(model, "module", model)
    state = getattr(unwrapped, "_npa_training_graphs", None)
    mode = configuration.get("npa_cuda_graphs", "off")
    if (state is not None) != (mode == "mot"):
        raise FlexPiError("loaded model differs from training CUDA graph policy")
    local = state.receipt() if state is not None else {"policy": "off"}
    local["rank"] = torch.distributed.get_rank()
    ranks = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(ranks, local)
    receipt = {"policy": mode, "ranks": ranks}
    validate_graphs_receipt(receipt, mode, len(ranks))
    return receipt


def validate_graphs_receipt(receipt, mode, world_size):
    """Require observed capture evidence from every requested training rank.

    Args:
        receipt: Worker phase's gathered graph evidence.
        mode: Requested off or mot policy.
        world_size: Number of participating training ranks.
    Returns:
        None.
    Raises:
        FlexPiError: A rank or required capture observation is absent or invalid.
    """
    if not isinstance(receipt, dict) or receipt.get("policy") != mode:
        raise FlexPiError("phase lacks the requested CUDA graph receipt")
    ranks = receipt.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise FlexPiError("CUDA graph receipt lacks every training rank")
    for rank, row in enumerate(ranks):
        if (
            not isinstance(row, dict)
            or type(row.get("rank")) is not int
            or row["rank"] != rank
            or row.get("policy") != mode
        ):
            raise FlexPiError("CUDA graph rank identity or policy differs")
        if mode == "mot":
            _validate_captured_rank(row)


def _validate_captured_rank(row):
    expected = {
        "backend": "make_graphed_callables",
        "fullgraph": True,
        "dynamic": False,
        "evaluation": "eager",
        "skipped_cuda_graphs": 0,
    }
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row[key] != value:
            raise FlexPiError("CUDA graph receipt differs from required backend")
    for key, minimum in (
        ("captured_training_calls", 2),
        ("eager_evaluation_calls", 0),
    ):
        if type(row.get(key)) is not int or row[key] < minimum:
            raise FlexPiError("CUDA graph receipt lacks observed training execution")
    replays = row.get("native_replays")
    if (
        not isinstance(replays, dict)
        or set(replays) != {"forward", "backward"}
        or any(
            type(count) is not int or count != row["captured_training_calls"]
            for count in replays.values()
        )
    ):
        raise FlexPiError(
            "CUDA graph receipt lacks matching native forward/backward replays"
        )
    recorded = row.get("recorded_cuda_graphs")
    if not isinstance(recorded, dict) or set(recorded) != {
        "forward",
        "backward",
        "inference",
    }:
        raise FlexPiError("CUDA graph receipt lacks observed capture modes")
    for mode, count in recorded.items():
        if type(count) is not int or (count != 0 if mode == "inference" else count < 1):
            raise FlexPiError("CUDA graph receipt lacks training-only capture")
