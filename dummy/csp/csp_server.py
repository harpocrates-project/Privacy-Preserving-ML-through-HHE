#!/usr/bin/env python
"""
csp_service.py

gRPC service that evaluates a DataMatrix file with a Keras binary‑desaturation
model and writes the predictions to a configurable folder.

The output file is named ``<prefix>_plaintext_binaryoutput.txt`` where
``<prefix>`` is everything before the first '_' in the original filename.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from io import StringIO
from pathlib import Path

import grpc
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------------------
# Generated protobuf / gRPC code (run protoc after editing csp.proto)
# ----------------------------------------------------------------------
import csp_pb2 as csp_pb2
import csp_pb2_grpc as csp_pb2_grpc

# ----------------------------------------------------------------------
# Configuration ---------------------------------------------------------
# ----------------------------------------------------------------------
DEFAULT_OUT_DIR = Path("binaryoutput")          # default folder under cwd
MODEL_PATH = Path("siesta_ml_binary.keras")    # model used for every request

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Keras model not found at {MODEL_PATH}")

# Load the model once – cheap compared with per‑request loading.
MODEL = tf.keras.models.load_model(str(MODEL_PATH))
logging.info("Loaded Keras model from %s", MODEL_PATH)


# ----------------------------------------------------------------------
# Helper functions (adapted from baseline.py)
# ----------------------------------------------------------------------
def _load_data_from_bytes(data_bytes: bytes) -> np.ndarray:
    """Parse a comma‑separated DataMatrix supplied as raw bytes (in‑memory)."""
    text = data_bytes.decode("utf-8")
    return np.loadtxt(StringIO(text), delimiter=",")


def _predict(data: np.ndarray) -> np.ndarray:
    """Run the binary‑desaturation model and return 0/1 predictions."""
    probs = MODEL.predict(data, verbose=0)          # shape (n_samples, 2)
    return np.argmax(probs, axis=1).astype(int)    # 1‑D array of 0/1


def _format_predictions(preds: np.ndarray) -> str:
    """Convert the integer array to the plain‑text BinaryOutput format."""
    return "\n".join(map(str, preds))


def _write_binary_output(pred_str: str, original_name: str, out_dir: Path) -> Path:
    """
    Write the prediction string to ``<out_dir>/<prefix>_plaintext_binaryoutput.txt``.

    Parameters
    ----------
    pred_str : str
        New‑line‑separated prediction string.
    original_name : str
        The filename the client supplied (may include a path).
    out_dir : pathlib.Path
        Directory where the output file will be placed.

    The ``prefix`` is everything **before the first '_'** in the *basename*.
    If the name contains no underscore the whole stem is used.

    Example
    -------
    client sends   filename = "DataMatrix_SpO2_cleaned4__esada.txt"
    → prefix = "DataMatrix"
    → output file = "<out_dir>/DataMatrix_plaintext_binaryoutput.txt"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep only the file name (drop any directory components)
    basename = Path(original_name).name          # e.g. "DataMatrix_SpO2_cleaned4__esada.txt"

    # Extract everything before the first underscore
    prefix = basename.split("_", 1)[0]

    out_path = out_dir / f"{prefix}_plaintext_binaryoutput.txt"
    out_path.write_text(pred_str + "\n")          # trailing newline matches original format
    return out_path


# ----------------------------------------------------------------------
# CSPService implementation
# ----------------------------------------------------------------------
class CSPServiceImpl(csp_pb2_grpc.CSPServiceServicer):
    """
    Implements the ``evaluate`` RPC.

    The request contains the raw DataMatrix bytes; the server processes them
    completely in memory and finally writes only the *prediction* file to disk.
    """

    def __init__(self, out_dir: Path):
        """
        Parameters
        ----------
        out_dir : pathlib.Path
            Directory where all BinaryOutput files will be stored.
        """
        self._out_dir = out_dir

    async def evaluate(
        self,
        request: csp_pb2.DataFile,
        context: grpc.aio.ServicerContext,
    ) -> csp_pb2.Empty:
        # --------------------------------------------------------------
        # 1️⃣  Basic validation
        # --------------------------------------------------------------
        if not request.data:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("DataFile.data must contain the file bytes")
            return csp_pb2.Empty()

        # ``filename`` is optional – we keep it for a nice output name.
        filename = request.filename or "uploaded_data_matrix.txt"
        logging.info(
            "Received evaluate request – %d bytes (filename=%s)",
            len(request.data),
            filename,
        )

        # --------------------------------------------------------------
        # 2️⃣  Load the matrix, run the model, format predictions
        # --------------------------------------------------------------
        try:
            data_matrix = _load_data_from_bytes(request.data)
            preds = _predict(data_matrix)
            pred_str = _format_predictions(preds)
        except Exception as exc:          # defensive
            logging.exception("Failed to generate predictions")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return csp_pb2.Empty()

        # --------------------------------------------------------------
        # 3️⃣  Write the BinaryOutput file to the configured folder
        # --------------------------------------------------------------
        try:
            out_path = _write_binary_output(pred_str, filename, self._out_dir)
            logging.info(
                "BinaryOutput written to %s (%d bytes)",
                out_path,
                out_path.stat().st_size,
            )
        except Exception as exc:          # defensive
            logging.exception("Failed to write BinaryOutput file")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return csp_pb2.Empty()

        # --------------------------------------------------------------
        # 4️⃣  Return an empty response (as defined in the proto)
        # --------------------------------------------------------------
        return csp_pb2.Empty()


# ----------------------------------------------------------------------
# Server start‑up
# ----------------------------------------------------------------------
async def serve(
    host: str = "[::]",
    port: int = 50051,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> None:
    """
    Starts the async gRPC server and registers CSPService.

    Parameters
    ----------
    host : str
        Interface to bind to (default ``[::]`` – all IPv4/IPv6).
    port : int
        Port number (default 50051).
    out_dir : pathlib.Path
        Directory where BinaryOutput files will be stored.
    """
    server = grpc.aio.server()
    csp_pb2_grpc.add_CSPServiceServicer_to_server(CSPServiceImpl(out_dir), server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logging.info(
        "CSPService listening on %s – output directory: %s",
        listen_addr,
        out_dir,
    )

    await server.start()
    await server.wait_for_termination()


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "gRPC CSPService – evaluate a DataMatrix file (sent as bytes) "
            "and write BinaryOutput files to a configurable folder."
        )
    )
    parser.add_argument(
        "--host",
        default="[::]",
        help="Interface to bind to (default: [::] – all IPv4/IPv6).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50052,
        help="Port number to listen on (default: 50052).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.getenv("CSP_OUT_DIR", str(DEFAULT_OUT_DIR)),
        help=(
            "Directory where BinaryOutput files will be stored. "
            "Can also be set via the CSP_OUT_DIR environment variable."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_cli()
    out_path = Path(args.out_dir).expanduser().resolve()
    try:
        asyncio.run(serve(host=args.host, port=args.port, out_dir=out_path))
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
