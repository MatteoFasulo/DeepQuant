"""
Custom model preparation script for quantization with DeepQuant and Brevitas.
Includes fixes for specific graph patterns in CCT and ViT models.

To date, after exporting the ONNX graph, the onnxruntime transformer optimizer must be run with:

python -m onnxruntime.transformers.optimizer --input Tests/ONNX/network.onnx --output Tests/ONNX/network.onnx --model_type vit --num_heads 3 --hidden_size 192 --use_multi_head_attention --disable_bias_gelu --disable_bias_skip_layer_norm --disable_skip_layer_norm --use_multi_head_attention --opt_level 0

And since sometimes some symbolic shapes are lost, the symbolic shape inference tool must be run as well:

python -m onnxruntime.tools.symbolic_shape_infer --input Tests/ONNX/network.onnx --output Tests/ONNX/network.onnx
"""

import argparse
import logging
import re
import sys
import time
from typing import Tuple

import brevitas.nn as qnn
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

# from brevitas.core.restrict_val import RestrictValueType
# from brevitas.core.scaling import ScalingImplType
from brevitas.fx.brevitas_tracer import symbolic_trace
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode
from brevitas.graph.quantize import preprocess_for_quantize, quantize

# from brevitas.inject.enum import StatsOp
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)
from tqdm import tqdm

from DeepQuant.Export import brevitasToTrueQuant
from DeepQuant.Transforms.Executor import TransformationExecutor
from DeepQuant.Transforms.Transformations import LinearTransformation, MHATransformation
from DeepQuant.Utils.ConsoleFormatter import ConsoleColor as cc
from DeepQuant.Utils.CustomTracer import QuantTracer, customBrevitasTrace
from DeepQuant.Utils.GraphPrinter import GraphModulePrinter
from emg_dataset import EMGDataset
from emg_transformer import EmgTransformer, MlpClassificationHead

logger = logging.getLogger("quant_logger")
handler = logging.StreamHandler(sys.stdout)  # stdout only
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)

QUANT_SCALE_PARAM = {
    # "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
    # "scaling_impl_type": ScalingImplType.AFFINE_STATS,
    # "scaling_stats_op": StatsOp.MAX,
}
CONV_BIAS = True  # Whether convolutional layers should have bias terms
LINEAR_BIAS = True  # Whether linear layers should have bias terms


def split_qkv_weight(weight: torch.Tensor):
    """
    Split a qkv weight into q,k,v according to shape.
    Returns tuple (q, k, v).
    Handles both common layouts:
      - concatenated along output dim: weight.shape[0] == 3 * hidden
      - concatenated along input dim:  weight.shape[1] == 3 * hidden  (we transpose to split)
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    # Case A: concatenated along output rows: (3*H, D)
    if out_dim % 3 == 0:
        h = out_dim // 3
        q = weight[0:h, :].clone()
        k = weight[h : 2 * h, :].clone()
        v = weight[2 * h : 3 * h, :].clone()
        return q, k, v
    # Case B: concatenated along input cols: (H, 3*D)  (less common)
    if in_dim % 3 == 0:
        h = in_dim // 3
        # transpose -> split -> transpose back
        wt_t = weight.t().contiguous()  # shape (in_dim, out_dim)
        q_t = wt_t[0:h, :].contiguous()
        k_t = wt_t[h : 2 * h, :].contiguous()
        v_t = wt_t[2 * h : 3 * h, :].contiguous()
        # transpose back to original orientation
        return q_t.t().contiguous(), k_t.t().contiguous(), v_t.t().contiguous()
    raise ValueError(
        f"Weight shape {weight.shape} is not a 3-way concatenation along rows or cols."
    )


def split_qkv_bias(bias: torch.Tensor):
    """
    Split qkv bias into q,k,v. bias must be 1D.
    """
    if bias is None:
        return None, None, None
    if bias.ndim != 1:
        raise ValueError(f"Expected 1D bias, got {tuple(bias.shape)}")
    n = bias.shape[0]
    if n % 3 == 0:
        h = n // 3
        return bias[0:h].clone(), bias[h : 2 * h].clone(), bias[2 * h : 3 * h].clone()
    raise ValueError(f"Bias length {n} is not divisible by 3.")


def convert_state_dict_qkv_to_qkv_separate(
    state_dict: dict,
    qkv_key_pattern=re.compile(r"(.*)\.qkv\.(weight|bias)$"),
    separate_template="{prefix}.q.{param}",
    key_replace_prefix=None,
):
    """
    Convert state_dict keys that contain '.qkv.weight' or '.qkv.bias' into
    separate '.q.weight', '.k.weight', '.v.weight' (and same for biases).
    - state_dict: original state dict (dict of tensors)
    - qkv_key_pattern: regex to find qkv keys. by default matches '<prefix>.qkv.weight' and '<prefix>.qkv.bias'
    - separate_template: how to name output keys (not commonly changed)
    - key_replace_prefix: optional function to alter prefix names (not required)
    Returns: new_state_dict
    """
    new_sd = {}
    handled_keys = set()

    for key, val in state_dict.items():
        m = qkv_key_pattern.match(key)
        if not m:
            # copy other params unchanged
            new_sd[key] = val
            continue

        prefix = m.group(1)  # part before '.qkv.weight' or '.qkv.bias'

        if key in handled_keys:
            continue

        # retrieve weight and bias (bias may be missing)
        weight_key = f"{prefix}.qkv.weight"
        bias_key = f"{prefix}.qkv.bias"

        weight = state_dict.get(weight_key, None)
        bias = state_dict.get(bias_key, None)

        if weight is None:
            raise KeyError(
                f"Expected weight at {weight_key} but it is missing in provided state_dict."
            )

        # split
        try:
            q_w, k_w, v_w = split_qkv_weight(weight)
        except Exception as e:
            raise RuntimeError(f"Error splitting {weight_key}: {e}")

        # split bias if present
        if bias is not None:
            try:
                q_b, k_b, v_b = split_qkv_bias(bias)
            except Exception as e:
                raise RuntimeError(f"Error splitting bias {bias_key}: {e}")
        else:
            q_b = k_b = v_b = None

        # build new keys and insert
        for name, w, b in (("q", q_w, q_b), ("k", k_w, k_b), ("v", v_w, v_b)):
            wkey = f"{prefix}.{name}.weight"  # user said they expect blocks.X.attn.q.weight etc.
            new_sd[wkey] = w
            if b is not None:
                bkey = f"{prefix}.{name}.bias"
                new_sd[bkey] = b

        handled_keys.add(weight_key)
        if bias is not None:
            handled_keys.add(bias_key)

    return new_sd


def describe_latency(latencies_seconds):
    arr = np.array(latencies_seconds)
    if arr.size == 0:
        return {}
    stats = {
        "count": int(arr.size),
        "mean_ms": float(arr.mean() * 1000.0),
        "median_ms": float(np.median(arr) * 1000.0),
        "p50_ms": float(np.percentile(arr, 50) * 1000.0),
        "p90_ms": float(np.percentile(arr, 90) * 1000.0),
        "p99_ms": float(np.percentile(arr, 99) * 1000.0),
        "throughput_samples_per_sec": (
            float(1.0 / arr.mean()) if arr.mean() > 0 else float("inf")
        ),
    }
    return stats


def evaluate_model_torch(
    model, dataloader, device="cpu", warmup_batches=5, name="model"
):
    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    latencies = []

    # Warm-up (optional)
    it = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(device)
        with torch.no_grad():
            _ = model(x.unsqueeze(1))

    # Real eval
    with torch.no_grad():
        for _, (inputs, targets) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc=f"Evaluating {name}"
        ):
            # inputs: (batch, channels?, ...) adjust if needed
            batch_size = inputs.shape[0]
            inputs = inputs.to(device)
            targets = targets.to(device)

            t0 = time.perf_counter()
            outputs = model(inputs.unsqueeze(1))
            t1 = time.perf_counter()
            elapsed = t1 - t0
            # per-sample latency
            per_sample = elapsed / float(batch_size)
            latencies.extend([per_sample] * batch_size)

            # handle model output shape
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += batch_size

    acc = float(total_correct) / float(total_samples) if total_samples > 0 else 0.0
    latency_stats = describe_latency(latencies)
    logger.info("%s accuracy: %.4f (%d samples)", name, acc, total_samples)
    logger.info("%s latency stats: %s", name, latency_stats)
    return acc, latency_stats


def evaluate_model_onnx(
    onnx_path, dataloader, device="cpu", warmup_batches=5, name="onnx"
):
    providers = ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name

    # Warm-up
    it = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        ort_inputs = {input_name: x.unsqueeze(1).numpy()}
        _ = ort_session.run(None, ort_inputs)

    total_correct = 0
    total_samples = 0
    latencies = []
    for _, (inputs, targets) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc=f"Evaluating {name} (ONNX)"
    ):
        batch_size = inputs.shape[0]
        ort_inputs = {input_name: inputs.unsqueeze(1).numpy()}
        t0 = time.perf_counter()
        outs = ort_session.run(None, ort_inputs)[0]
        t1 = time.perf_counter()
        elapsed = t1 - t0
        per_sample = elapsed / float(batch_size)
        latencies.extend([per_sample] * batch_size)

        out_t = torch.from_numpy(outs)
        if out_t.dim() > 1:
            _, predicted = torch.max(out_t.data, 1)
        else:
            predicted = (out_t > 0.5).long().squeeze()

        targets = targets.view_as(predicted)
        total_correct += (predicted == targets).sum().item()
        total_samples += batch_size

    acc = float(total_correct) / float(total_samples) if total_samples > 0 else 0.0
    latency_stats = describe_latency(latencies)
    logger.info("%s (ONNX) accuracy: %.4f (%d samples)", name, acc, total_samples)
    logger.info("%s (ONNX) latency stats: %s", name, latency_stats)
    return acc, latency_stats


def injectCustomForwards(
    model: nn.Module,
    exampleInput: torch.Tensor,
    referenceOutput: torch.Tensor,
    debug: bool = False,
    checkEquivalence: bool = False,
) -> Tuple[nn.Module, torch.Tensor]:
    """Custom inject function for CCT that excludes ActivationTransformation."""
    printer = GraphModulePrinter()

    tracer = QuantTracer(debug=debug)

    transformations = [
        MHATransformation(),
        LinearTransformation(),
        # ActivationTransformation(),  # FBRANCASI: Commented out for CCT compatibility
    ]

    executor = TransformationExecutor(transformations, debug=debug, tracer=tracer)
    transformedModel = executor.execute(model, exampleInput)

    fxModel = customBrevitasTrace(
        root=transformedModel,
        tracer=tracer,
    )
    fxModel.recompile()

    with torch.no_grad():
        output = fxModel(exampleInput)

    if checkEquivalence:
        if torch.allclose(referenceOutput, output, atol=1e-5):
            if debug:
                print(cc.success("Injection of New Modules: output is consistent"))
        else:
            raise RuntimeError(
                cc.error("Injection of New Modules changed the output significantly")
            )

    if debug:
        print(cc.header("2. Network after Injection of New Modules"))
        printer.printTabular(fxModel)
        print()

    return fxModel, output


def replace_all_uses_except(old_node, new_node, exceptions):
    """Replace all uses of old_node with new_node, except for nodes in exceptions list."""
    for user in list(old_node.users):
        if user not in exceptions:
            user.replace_input_with(old_node, new_node)

    return new_node


def find_transpose_add(model, verbose: bool = False) -> list:
    """Find patterns of transpose followed by add in the model graph."""
    patterns = []
    for node in model.graph.nodes:
        if node.op == "call_method" and node.target == "transpose":
            for user in node.users:
                if (
                    "add" in user.name
                    or user.target in [torch.add]
                    or (user.op == "call_method" and user.target in ["add", "add_"])
                ):
                    patterns.append((node, user))
                    logger.debug(
                        "Found transpose-add pattern: %s -> %s",
                        node.name,
                        user.name,
                    )
                    break
    return patterns


def find_qkv_reshape(model, verbose: bool = False) -> list:
    """Find patterns of QKV projection followed by reshape in the model graph."""
    patterns = []
    for node in model.graph.nodes:
        if node.op == "call_module" and "qkv" in node.target:
            for user in node.users:
                if user.op == "call_method" and user.target == "reshape":
                    patterns.append((node, user))
                    logger.debug(
                        "Found QKV-reshape pattern: %s -> %s", node.name, user.name
                    )
                    break
    return patterns


def find_matmul_dequantize(model, verbose: bool = False) -> list:
    """Find matmul nodes that may need dequantization."""
    matmul_nodes = []
    for node in model.graph.nodes:
        if node.op == "call_function" and node.target == torch.matmul:
            matmul_nodes.append(node)
            logger.debug(f"Found matmul node: {node.name}")
        elif node.op == "call_method" and node.target == "matmul":
            matmul_nodes.append(node)
            logger.debug(f"Found matmul node: {node.name}")
        elif (
            node.op == "call_function"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "matmul"
        ):
            matmul_nodes.append(node)
            logger.debug(f"Found matmul node: {node.name}")
        elif hasattr(node, "name") and "matmul" in node.name:
            matmul_nodes.append(node)
            logger.debug(f"Found matmul node: {node.name}")
        elif (
            node.op == "call_function"
            and hasattr(node.target, "__module__")
            and node.target.__module__ == "operator"
            and hasattr(node.target, "__name__")
            and node.target.__name__ == "matmul"
        ):
            matmul_nodes.append(node)
            logger.debug(f"Found matmul node: {node.name}")
    return matmul_nodes


def apply_transpose_fix(model, node, user):
    """Apply transpose fix by inserting QuantIdentity before the add operation."""
    quant_identity = qnn.QuantIdentity(
        act_quant=Int8ActPerTensorFloat,
        return_quant_tensor=True,
        **QUANT_SCALE_PARAM,
    )

    quant_name = f"{node.name}_quant_fix"
    model.add_module(quant_name, quant_identity)

    with model.graph.inserting_after(node):
        quant_node = model.graph.call_module(quant_name, args=(node,))

    # Replace uses
    replace_all_uses_except(node, quant_node, [quant_node])

    return quant_node


def apply_qkv_fix(model, node, reshape_user):
    """Apply QKV fix by inserting QuantIdentity before the reshape operation."""
    quant_identity = qnn.QuantIdentity(
        act_quant=Int8ActPerTensorFloat,
        return_quant_tensor=False,  # Return regular tensor for reshape
        **QUANT_SCALE_PARAM,
    )

    quant_name = f"{node.name}_reshape_fix"
    model.add_module(quant_name, quant_identity)

    with model.graph.inserting_after(node):
        quant_node = model.graph.call_module(quant_name, args=(node,))

    reshape_user.update_arg(0, quant_node)

    return quant_node


# keep a module-level cache to avoid duplicate dequant nodes
_dequant_cache = {}


def apply_matmul_fix(
    model: torch.fx.GraphModule,
    producer_node: torch.fx.Node,
    matmul_node: torch.fx.Node,
    arg_index: int,
):
    """
    Insert QuantIdentity after producer_node, then update matmul_node.args[arg_index]
    to use the new dequant node. Reuse dequant node for same producer if previously created.
    """
    global _dequant_cache
    cache_key = producer_node.name

    # If we've already created a dequant for this producer, reuse it
    if cache_key in _dequant_cache:
        dequant_node = _dequant_cache[cache_key]
    else:
        quant_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
            **QUANT_SCALE_PARAM,
        )
        quant_name = f"{producer_node.name}_dequant_for_matmul"
        # ensure uniqueness in module names
        unique_name = quant_name
        idx = 0
        while hasattr(model, unique_name):
            idx += 1
            unique_name = f"{quant_name}_{idx}"
        model.add_module(unique_name, quant_identity)

        # Insert after producer_node so topological order is preserved
        with model.graph.inserting_after(producer_node):
            dequant_node = model.graph.call_module(unique_name, args=(producer_node,))

        # cache it
        _dequant_cache[cache_key] = dequant_node

    # Replace the specific argument of the matmul node
    new_args = list(matmul_node.args)
    new_args[arg_index] = dequant_node
    matmul_node.args = tuple(new_args)

    return dequant_node


def prepare_my_model(model, verbose: bool = False) -> nn.Module:
    model = model.eval()

    if not hasattr(model, "graph"):
        model = symbolic_trace(model)

    logger.debug("=== FIXING QUANTIZATION ISSUES ===")
    transpose_fixes, qkv_fixes, matmul_fixes = [], [], []

    # FBRANCASI: Fix 1, Find transpose -> add patterns
    transpose_fixes = find_transpose_add(model, verbose=verbose)
    # FBRANCASI: Fix 2, Find QKV -> reshape patterns
    qkv_fixes = find_qkv_reshape(model, verbose=verbose)
    # FBRANCASI: Fix 3, Find matmul operations that need dequantization
    matmul_fixes = find_matmul_dequantize(model, verbose=verbose)

    print("\n=== APPLYING GRAPH MODIFICATIONS ===")

    # FBRANCASI: Apply transpose fixes
    print(f"Applying {len(transpose_fixes)} transpose fixes...")
    for node, user in transpose_fixes:
        logger.debug(f"  Fixing: {node.name} -> {user.name}")
        apply_transpose_fix(model, node, user)

    # FBRANCASI: Apply QKV fixes
    print(f"Applying {len(qkv_fixes)} QKV fixes...")
    for node, reshape_user in qkv_fixes:
        logger.debug(f"  Fixing: {node.name} -> {reshape_user.name}")
        apply_qkv_fix(model, node, reshape_user)

    # FBRANCASI: Apply matmul fixes
    print(f"Applying {len(matmul_fixes)} matmul fixes...")
    for matmul_node in matmul_fixes:
        logger.debug(
            f"Fixing matmul: {matmul_node.name}; args = {[getattr(a,'name',str(a)) for a in matmul_node.args]}"
        )
        for i, arg in enumerate(list(matmul_node.args)):  # <-- list() to copy
            if isinstance(arg, torch.fx.Node):
                # skip if this arg is already a QuantIdentity/dequant we inserted earlier
                if "dequant_for_matmul" in getattr(arg, "name", ""):
                    logger.debug(
                        f"  skipping arg {i} ({arg.name}) — already dequantized"
                    )
                    continue
                # Insert dequant after the producer (arg) and update matmul arg
                apply_matmul_fix(
                    model=model, producer_node=arg, matmul_node=matmul_node, arg_index=i
                )

    model.recompile()
    model.graph.lint()

    logger.debug("\n=== GRAPH MODIFICATION COMPLETE ===")
    # Debug: Print graph structure to understand the flow
    logger.debug("\n=== DEBUG: Graph structure after fixes ===")
    for node in model.graph.nodes:
        if (
            "matmul" in node.name
            or (node.op == "call_method" and node.target == "transpose")
            or "permute" in node.name
        ):
            logger.debug(
                f"Node: {node.name}, op: {node.op}, target: {node.target}, args: {[arg.name if hasattr(arg, 'name') else str(arg) for arg in node.args]}"
            )
            # Print users of permute and transpose nodes
            if "permute" in node.name or (
                node.op == "call_method" and node.target == "transpose"
            ):
                logger.debug(f"  Users: {[user.name for user in node.users]}")

    # FBRANCASI: First pass - identify which Linear layers feed into matmul through permute/transpose
    linear_to_matmul = set()
    for node in model.graph.nodes:
        if hasattr(node, "name") and "matmul" in node.name:
            # Trace back through the args to find Linear layers
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    # Check if this path leads back to a linear layer
                    current = arg
                    visited = set()
                    while current and current not in visited:
                        visited.add(current)
                        if current.op == "call_module" and any(
                            proj in current.target
                            for proj in ["q_proj", "k_proj", "v_proj"]
                        ):
                            linear_to_matmul.add(current.target)
                            break
                        # Trace back through the first argument
                        if current.args and isinstance(current.args[0], torch.fx.Node):
                            current = current.args[0]
                        else:
                            break

    logger.debug("\nLinear layers that feed into matmul: %s", linear_to_matmul)

    compute_layer_map = {
        nn.Conv2d: (
            qnn.QuantConv2d,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "bias": CONV_BIAS,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                **QUANT_SCALE_PARAM,
            },
        ),
        nn.Linear: (
            qnn.QuantLinear,
            {
                "input_quant": Int8ActPerTensorFloat,
                "weight_quant": Int8WeightPerTensorFloat,
                "output_quant": Int8ActPerTensorFloat,
                "bias_quant": Int32Bias,
                "bias": LINEAR_BIAS,
                "return_quant_tensor": True,
                "output_bit_width": 8,
                **QUANT_SCALE_PARAM,
            },
        ),
    }

    quant_act_map = {
        # nn.ReLU: (
        #    qnn.QuantReLU,
        #    {
        #        "act_quant": Int8ActPerTensorFloat,
        #        "return_quant_tensor": True,
        #        "bit_width": 8,
        #        **QUANT_SCALE_PARAM,
        #    },
        # ),
    }

    quant_identity_map = {
        "signed": (
            qnn.QuantIdentity,
            {
                "act_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
                **QUANT_SCALE_PARAM,
            },
        ),
        "unsigned": (
            qnn.QuantIdentity,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
                **QUANT_SCALE_PARAM,
            },
        ),
    }

    logger.debug("\nPreprocessing model for quantization...")
    model = preprocess_for_quantize(
        model,
        trace_model=False,  # FBRANCASI: Already traced
        equalize_iters=0,
        equalize_scale_computation="range",
        equalize_merge_bias=False,
    )

    logger.debug("Model preprocessing complete.")
    logger.debug("\nQuantizing model...")
    quantized_model = quantize(
        graph_model=model,
        quant_identity_map=quant_identity_map,
        compute_layer_map=compute_layer_map,
        quant_act_map=quant_act_map,
    )

    # FBRANCASI: Apply post-quantization fixes for matmul operations
    logger.debug("\n=== POST-QUANTIZATION FIXES ===")

    nodes_needing_dequant = set()

    node_map = {node.name: node for node in quantized_model.graph.nodes}

    import operator

    for node in quantized_model.graph.nodes:
        # FBRANCASI: Look for @ operator (represented as call_function with operator.matmul)
        is_matmul = False
        if node.op == "call_function":
            if node.target == operator.matmul:
                is_matmul = True
            elif node.target == torch.matmul:
                is_matmul = True
            elif hasattr(node.target, "__name__") and node.target.__name__ == "matmul":
                is_matmul = True

        if is_matmul:
            logger.debug(f"\nFound matmul node: {node.name}")
            logger.debug(f"  Target: {node.target}")
            logger.debug(f"  Args: {node.args}")
            logger.debug(f"  Arg types: {[type(arg) for arg in node.args]}")

            # FBRANCASI: Mark both arguments as needing dequantization
            for i, arg in enumerate(node.args):
                logger.debug(f"    Checking arg {i}: type={type(arg)}")
                if hasattr(arg, "name") and hasattr(arg, "op"):
                    nodes_needing_dequant.add(arg)
                    logger.debug(f"    Added node to dequant: {arg.name}")
                else:
                    logger.debug(f"    Skipped arg {i}: {arg}")

    logger.debug(
        f"\nNodes needing dequantization: {[n.name for n in nodes_needing_dequant]}"
    )

    # FBRANCASI: Insert dequantization for each node that feeds into matmul
    dequant_nodes = {}
    for node in nodes_needing_dequant:
        logger.debug(f"\nAdding dequantization after node: {node.name}")

        dequant_identity = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,  # FBRANCASI: Return regular tensor for matmul
            **QUANT_SCALE_PARAM,
        )

        dequant_name = f"{node.name}_dequant_for_matmul"
        quantized_model.add_module(dequant_name, dequant_identity)

        with quantized_model.graph.inserting_after(node):
            dequant_node = quantized_model.graph.call_module(dequant_name, args=(node,))

        dequant_nodes[node] = dequant_node

        for user in list(node.users):
            is_matmul_user = False
            if user.op == "call_function":
                if user.target == operator.matmul or user.target == torch.matmul:
                    is_matmul_user = True
                elif (
                    hasattr(user.target, "__name__")
                    and user.target.__name__ == "matmul"
                ):
                    is_matmul_user = True
                elif (
                    hasattr(user.target, "__module__")
                    and user.target.__module__ == "operator"
                    and hasattr(user.target, "__name__")
                    and user.target.__name__ == "matmul"
                ):
                    is_matmul_user = True

            if is_matmul_user:
                logger.debug(f"  Updating matmul {user.name} to use dequantized input")
                new_args = []
                for i, arg in enumerate(user.args):
                    if arg == node:
                        new_args.append(dequant_node)
                        logger.debug(
                            f"    Updated arg {i} from {node.name} to {dequant_node.name}"
                        )
                    else:
                        new_args.append(arg)
                user.args = tuple(new_args)

    quantized_model.recompile()
    quantized_model.graph.lint()

    logger.debug("\n=== POST-QUANTIZATION FIXES COMPLETE ===")

    return quantized_model


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser(description="Test model quantization")
    argumentParser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    argumentParser.add_argument(
        "--ckpt",
        type=str,
        default="/home/matteo/Scaricati/new_uci_emg_pretrained_full_finetune-epoch=08-val_loss=0.9158.ckpt",
        help="Path to checkpoint file",
    )
    argumentParser.add_argument(
        "--calibration_data",
        type=str,
        default="/home/matteo/Scaricati/val.h5",
        help="Path to calibration data file",
    )
    argumentParser.add_argument(
        "--test_data",
        type=str,
        default="/home/matteo/Scaricati/test.h5",
        help="Path to test data file",
    )
    argumentParser.add_argument(
        "--apply_calib",
        action="store_true",
        help="Apply calibration to the model",
    )
    argumentParser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    args = argumentParser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    device = "cpu"
    torch.manual_seed(args.seed)
    sample_input = torch.randn(1, 1, 8, 1000).to(device)  # Example input tensor

    # build model
    encoder = EmgTransformer(
        img_size=1000,
        patch_size=20,
        in_chans=16,
        embed_dim=192,
        n_layer=2,
        n_head=3,
        mlp_ratio=4,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_bias=CONV_BIAS,
    )
    model_head = MlpClassificationHead(
        embed_dim=192,
        num_classes=7,
        reduction="concat",
        in_chans=16,
        bias=LINEAR_BIAS,
    )

    # load weights
    # weights = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # state_dict = weights["state_dict"]
    # logger.info("Loaded state dict keys: %d", len(state_dict.keys()))
    #
    # pretrained_params = {
    #    k.replace("model.", ""): v
    #    for k, v in state_dict.items()
    #    if k.startswith("model.")
    # }
    # pretrained_params = {k.replace("patch_embed.", ""): v for k, v in pretrained_params.items()}
    #
    ## Convert QKV shared weights to separate Q,K,V weights
    # pretrained_params = convert_state_dict_qkv_to_qkv_separate(pretrained_params)
    # encoder.load_state_dict(pretrained_params, strict=True if encoder.n_layer == 8 else False)
    # model_head.load_state_dict(
    #    {k.replace("model_head.", ""): v for k, v in state_dict.items() if k.startswith("model_head.")},
    #    strict=True,
    # )

    encoder.eval()
    model_head.eval()

    class FullModel(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            x = self.encoder(x)
            x = self.head(x)
            return x

    model = FullModel(encoder, model_head)
    model.to(device)
    model.eval()

    # Prepare dataloaders
    # calib_dset = EMGDataset(args.calibration_data, finetune=True)
    # calib_loader = torch.utils.data.DataLoader(
    #    calib_dset, batch_size=32, shuffle=False, pin_memory=True
    # )
    #
    # test_dset = EMGDataset(args.test_data, finetune=True)
    # test_loader = torch.utils.data.DataLoader(
    #    test_dset, batch_size=1, shuffle=False, pin_memory=True
    # )

    # Export FP32 ONNX for reference
    onnx_fp32_path = "Tests/ONNX/model.onnx"
    torch.onnx.export(model, (sample_input,), onnx_fp32_path, opset_version=17)
    logger.info("Exported FP32 ONNX to %s", onnx_fp32_path)

    # FP32 evaluation (ONNXRuntime)
    if args.verbose:
        logger.info("Evaluating FP32 (ONNXRuntime) model...")
        fp32_acc, fp32_latency = evaluate_model_onnx(
            onnx_fp32_path,
            test_loader,
            device=device,
            warmup_batches=5,
            name="FP32-ONNX",
        )

    # Prepare quantized model
    logger.info("Preparing quantized model...")
    quantized_model = prepare_my_model(model)

    # Calibration
    if args.apply_calib:
        logger.info("Calibrating quantized model (PTQ) ...")
        with torch.no_grad():
            with calibration_mode(quantized_model):
                for x, _ in tqdm(
                    calib_loader, desc="Calibrating", total=len(calib_loader)
                ):
                    x = x.unsqueeze(1).to(device)
                    quantized_model(x)

            # Apply bias correction if available
            with bias_correction_mode(quantized_model):
                for x, _ in tqdm(
                    calib_loader, desc="Bias Correction", total=len(calib_loader)
                ):
                    x = x.unsqueeze(1).to(device)
                    quantized_model(x)

    # FBRANCASI: Override the injectCustomForwards function in the module before DeepQuant.Export imports it
    import DeepQuant.Pipeline.Injection as injection_module

    # FBRANCASI: Store original function
    original_inject = injection_module.injectCustomForwards

    # FBRANCASI: Override with our custom function
    injection_module.injectCustomForwards = injectCustomForwards

    # FBRANCASI: Force reload of Export module to pick up the override
    import importlib

    import DeepQuant.Export

    importlib.reload(DeepQuant.Export)

    try:
        from DeepQuant.Export import brevitasToTrueQuant

        quantized_model.eval()
        brevitasToTrueQuant(quantized_model, sample_input, debug=args.verbose)
    finally:
        # FBRANCASI: Restore original function and reload Export module again
        injection_module.injectCustomForwards = original_inject
        importlib.reload(DeepQuant.Export)
        importlib.reload(DeepQuant.Export)

    if args.verbose:
        # INT8 evaluation (ONNXRuntime)
        onnx_int8_path = "Tests/ONNX/network.onnx"
        logger.info(
            "Evaluating ONNX model (path=%s) using ONNXRuntime...", onnx_int8_path
        )
        int8_acc_onnx, int8_latency_onnx = evaluate_model_onnx(
            onnx_int8_path,
            test_loader,
            device=device,
            warmup_batches=5,
            name="INT8-ONNX",
        )

        # Summary
        print("\n\n" + "=" * 90)
        print("{:^90}".format("EVALUATION SUMMARY"))
        print("=" * 90)
        print(
            "{:<22} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}".format(
                "Model (Backend)",
                "Accuracy",
                "Mean (ms)",
                "P90 (ms)",
                "P99 (ms)",
                "Throughput",
            )
        )
        print("-" * 90)

        def fmt_latency(lat):
            return (
                f"{lat['mean_ms']:.2f}",
                f"{lat['p90_ms']:.2f}",
                f"{lat['p99_ms']:.2f}",
                f"{lat['throughput_samples_per_sec']:.1f}",
            )

        fp32_mean, fp32_p90, fp32_p99, fp32_thr = fmt_latency(fp32_latency)
        int8_mean, int8_p90, int8_p99, int8_thr = fmt_latency(int8_latency_onnx)

        print(
            "{:<22} | {:<10.4f} | {:<10} | {:<10} | {:<10} | {:<10}".format(
                "FP32 (ONNXRuntime)", fp32_acc, fp32_mean, fp32_p90, fp32_p99, fp32_thr
            )
        )
        print(
            "{:<22} | {:<10.4f} | {:<10} | {:<10} | {:<10} | {:<10}".format(
                "INT8 (ONNXRuntime)",
                int8_acc_onnx,
                int8_mean,
                int8_p90,
                int8_p99,
                int8_thr,
            )
        )
        print("=" * 90 + "\n\n")

    # Optionally, print per-output debug differences on a single sample (like you did before)
    with torch.no_grad():
        ref_out = model(sample_input)
        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]
        q_out = quantized_model(sample_input)
        if isinstance(q_out, tuple):
            q_out = q_out[0]
        ref_out, q_out = ref_out.flatten(), q_out.flatten()
        for i in range(ref_out.shape[0]):
            logger.info(
                "Target: %.6f\tQuant: %.6f\tDiff: %.6f",
                ref_out[i].item(),
                q_out[i].item(),
                (ref_out[i] - q_out[i]).item(),
            )
