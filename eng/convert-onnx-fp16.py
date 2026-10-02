#!/usr/bin/env python3
"""Convert an ONNX decoder graph from float32 tensors/weights to float16."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert float32 tensors and initializers in an ONNX decoder graph "
            "to float16 using onnxconverter-common."
        )
    )
    parser.add_argument("--input", required=True, help="Source FP32 ONNX model.")
    parser.add_argument("--output", required=True, help="Destination FP16 ONNX model.")
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Store tensors in one external .data file next to the output model.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help=(
            "Skip ONNX checker for ORT-optimized graphs with nonstandard "
            "operators; validate with the target runtime."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Source ONNX model does not exist: {source}")
    if source == destination:
        raise ValueError("--output must differ from --input.")

    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError(
            "onnxconverter-common is required. Install it with "
            "'python -m pip install onnxconverter-common'."
        ) from exc

    model = onnx.load_model(source, load_external_data=True)

    before_initializers = sum(
        1 for initializer in model.graph.initializer
        if initializer.data_type == TensorProto.FLOAT
    )
    before_float_io = sum(
        1
        for value in [*model.graph.input, *model.graph.output]
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type == TensorProto.FLOAT
    )

    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=True,
    )

    remaining_float_initializers = [
        initializer.name
        for initializer in converted.graph.initializer
        if initializer.data_type == TensorProto.FLOAT
    ]
    remaining_float_io = [
        value.name
        for value in [*converted.graph.input, *converted.graph.output]
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type == TensorProto.FLOAT
    ]

    if remaining_float_io:
        preview = ", ".join(remaining_float_io[:8])
        raise ValueError(
            "FP16 conversion left float32 graph inputs/outputs: "
            f"{preview}"
        )

    if not args.skip_check:
        onnx.checker.check_model(converted)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.external_data:
        onnx.save_model(
            converted,
            destination,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=destination.name + ".data",
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save_model(converted, destination)

    print(
        f"Converted {source} -> {destination}; "
        f"float32 initializers before={before_initializers}, "
        f"float32 initializers remaining={len(remaining_float_initializers)}, "
        f"float32 graph IO converted={before_float_io}."
    )


if __name__ == "__main__":
    main()
