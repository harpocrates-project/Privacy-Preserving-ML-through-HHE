#!/usr/bin/env python
"""
csp_client.py

A tiny client that reads a DataMatrix file from disk, sends its *content*
(and the original filename) to the CSPService.evaluate RPC.
"""

import argparse
import asyncio
from pathlib import Path

import grpc
import csp_pb2 as csp_pb2
import csp_pb2_grpc as csp_pb2_grpc


async def call_evaluate(host: str, port: int, data_path: Path):
    """Read the file, send its bytes to the server, and wait for completion."""
    if not data_path.is_file():
        raise FileNotFoundError(f"DataMatrix file not found: {data_path}")

    # Read the whole file as bytes (UTF‑8 text, but we keep it as raw bytes)
    file_bytes = data_path.read_bytes()

    request = csp_pb2.DataFile(
        filename=data_path.name,   # optional – helps the server name the output file
        data=file_bytes,
    )

    target = f"{host}:{port}"
    async with grpc.aio.insecure_channel(target) as channel:
        stub = csp_pb2_grpc.CSPServiceStub(channel)
        await stub.evaluate(request)
        print(f"Sent {data_path} ({len(file_bytes)} bytes) – server will write predictions to disk.")


def main():
    parser = argparse.ArgumentParser(description="Send a DataMatrix file to CSPService.evaluate")
    parser.add_argument("--host", required=True, help="Server host (e.g. localhost)")
    parser.add_argument("--port", type=int, required=True, help="Server port (e.g. 50051)")
    parser.add_argument(
        "--datafile",
        type=Path,
        required=True,
        help="Path to the DataMatrix text file (comma‑separated)",
    )
    args = parser.parse_args()

    asyncio.run(call_evaluate(args.host, args.port, args.datafile))


if __name__ == "__main__":
    main()
