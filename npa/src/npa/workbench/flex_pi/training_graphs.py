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
    """Install a training-only compiled forward before distributed wrapping.

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
    if any(_recorded_graphs().values()):
        raise FlexPiError("training capture requires an unused CUDA graph manager")
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


def _compiler_counters():
    from torch._dynamo.utils import counters

    return {
        "skipped_cuda_graphs": counters["inductor"]["cudagraph_skips"],
        "compiled_graphs": counters["stats"]["unique_graphs"],
    }


def _recorded_graphs():
    import torch
    from torch._inductor.cudagraph_trees import get_manager

    # This internal tree layout is bound to the exact vendor version above.
    manager = get_manager(torch.cuda.current_device(), create_if_none_exists=False)
    counts = {"forward": 0, "backward": 0, "inference": 0}
    if manager is None:
        return counts
    pending = [(key, node) for key, roots in manager.roots.items() for node in roots]
    seen = set()
    while pending:
        key, node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        mode = manager.id_to_mode[key].name.lower()
        counts[mode] += int(node.graph is not None)
        pending.extend(
            (key, child)
            for key, children in node.children.items()
            for child in children
        )
    return counts


class _TrainingGraphs:
    def __init__(self, module):
        import torch

        self.module = module
        self.original = module.forward
        self.initial_counters = _compiler_counters()
        self.training_calls = 0
        self.evaluation_calls = 0
        self.compiled = torch.compile(
            self.original, backend="cudagraphs", fullgraph=True, dynamic=False
        )

    def check(self):
        if (
            _compiler_counters()["skipped_cuda_graphs"]
            != self.initial_counters["skipped_cuda_graphs"]
        ):
            raise FlexPiError(
                "training CUDA graph capture fell back to eager execution"
            )

    def forward(self, *args, **kwargs):
        import torch

        self.check()
        # Evaluation keeps the exact eager path; only training needs capture.
        if not self.module.training or not torch.is_grad_enabled():
            self.evaluation_calls += 1
            return self.original(*args, **kwargs)
        result = self.compiled(*args, **kwargs)
        self.training_calls += 1
        self.check()
        return result

    def receipt(self):
        self.check()
        recorded = _recorded_graphs()
        compiled = (
            _compiler_counters()["compiled_graphs"]
            - self.initial_counters["compiled_graphs"]
        )
        if (
            self.training_calls < 2
            or compiled < 1
            or recorded["forward"] < 1
            or recorded["backward"] < 1
        ):
            raise FlexPiError("training lacks recorded forward/backward CUDA graphs")
        return {
            "policy": "mot",
            "backend": "cudagraphs",
            "fullgraph": True,
            "dynamic": False,
            "evaluation": "eager",
            "compiled_training_calls": self.training_calls,
            "eager_evaluation_calls": self.evaluation_calls,
            "compiled_graphs": compiled,
            "recorded_cuda_graphs": recorded,
            "skipped_cuda_graphs": 0,
        }


def check_training_graphs(model):
    """Reject a compiler fallback observed during either forward or backward.

    Args:
        model: Original model, or its native DDP wrapper.
    Returns:
        None; eager models require no compiler inspection.
    Raises:
        FlexPiError: The requested capture fell back to eager execution.
    """
    unwrapped = getattr(model, "module", model)
    state = getattr(unwrapped, "_npa_training_graphs", None)
    if state is not None:
        state.check()


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
        "backend": "cudagraphs",
        "fullgraph": True,
        "dynamic": False,
        "evaluation": "eager",
        "skipped_cuda_graphs": 0,
    }
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row[key] != value:
            raise FlexPiError("CUDA graph receipt differs from required backend")
    for key, minimum in (
        ("compiled_training_calls", 2),
        ("eager_evaluation_calls", 0),
        ("compiled_graphs", 1),
    ):
        if type(row.get(key)) is not int or row[key] < minimum:
            raise FlexPiError("CUDA graph receipt lacks observed compiler execution")
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
