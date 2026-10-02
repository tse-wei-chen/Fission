#!/usr/bin/env python3
"""Append a graph-side greedy sampled-token output to a decoder ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append Gather(last sequence position) + ArgMax(vocabulary) to an "
            "ONNX decoder graph and expose the sampled token ids as [batch, 1]."
        )
    )
    parser.add_argument("--input", required=True, help="Source decoder ONNX model.")
    parser.add_argument("--output", required=True, help="Destination ONNX model.")
    parser.add_argument(
        "--logits-output",
        default="logits",
        help="Existing rank-3 decoder logits graph output name.",
    )
    parser.add_argument(
        "--sampled-output",
        default="fission_sampled_token_ids",
        help="New int64 [batch,1] graph output name.",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Store tensors in one external .data file next to the output model.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip ONNX checker for ORT-optimized graphs with nonstandard operators; validate with the target runtime.",
    )
    return parser.parse_args()


def collect_tensor_names(graph: onnx.GraphProto) -> set[str]:
    names: set[str] = set()
    names.update(value.name for value in graph.input)
    names.update(value.name for value in graph.output)
    names.update(value.name for value in graph.value_info)
    names.update(initializer.name for initializer in graph.initializer)
    for node in graph.node:
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)
    return names


def unique_name(existing: set[str], base: str) -> str:
    if base not in existing:
        existing.add(base)
        return base

    index = 1
    while f"{base}_{index}" in existing:
        index += 1
    name = f"{base}_{index}"
    existing.add(name)
    return name


def main() -> None:
    args = parse_args()
    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Source ONNX model does not exist: {source}")
    if source == destination:
        raise ValueError("--output must differ from --input.")
    if not args.logits_output.strip():
        raise ValueError("--logits-output cannot be empty.")
    if not args.sampled_output.strip():
        raise ValueError("--sampled-output cannot be empty.")

    model = onnx.load_model(source, load_external_data=True)
    graph = model.graph

    graph_outputs = {value.name: value for value in graph.output}
    if args.logits_output not in graph_outputs:
        available = ", ".join(sorted(graph_outputs))
        raise ValueError(
            f"Graph output '{args.logits_output}' was not found. "
            f"Available outputs: {available}"
        )

    logits_type = graph_outputs[args.logits_output].type.tensor_type
    if logits_type.elem_type not in (
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.DOUBLE,
    ):
        raise ValueError(
            f"Logits output '{args.logits_output}' must be floating point."
        )
    if len(logits_type.shape.dim) != 3:
        raise ValueError(
            f"Logits output '{args.logits_output}' must have rank 3."
        )

    existing_names = collect_tensor_names(graph)
    if args.sampled_output in existing_names:
        raise ValueError(
            f"Requested sampled output name '{args.sampled_output}' already exists."
        )
    existing_names.add(args.sampled_output)

    last_position_indices = unique_name(
        existing_names,
        "fission_last_sequence_position",
    )
    last_position_logits = unique_name(
        existing_names,
        "fission_last_position_logits",
    )

    graph.initializer.append(
        helper.make_tensor(
            name=last_position_indices,
            data_type=TensorProto.INT64,
            dims=[1],
            vals=[-1],
        )
    )
    graph.node.append(
        helper.make_node(
            "Gather",
            inputs=[args.logits_output, last_position_indices],
            outputs=[last_position_logits],
            axis=1,
            name="FissionGatherLastSequencePosition",
        )
    )
    graph.node.append(
        helper.make_node(
            "ArgMax",
            inputs=[last_position_logits],
            outputs=[args.sampled_output],
            axis=2,
            keepdims=0,
            name="FissionGreedyArgMax",
        )
    )
    graph.output.append(
        helper.make_tensor_value_info(
            args.sampled_output,
            TensorProto.INT64,
            ["batch", 1],
        )
    )

    if not args.skip_check:
        onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if args.external_data:
        data_name = destination.name + ".data"
        onnx.save_model(
            model,
            destination,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save_model(model, destination)

    print(
        f"Added graph-side greedy output '{args.sampled_output}' "
        f"to {destination}"
    )


if __name__ == "__main__":
    main()
