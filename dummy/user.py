import argparse
import grpc
import os

import csp_pb2
import csp_pb2_grpc

def run(address: str, filepath: str):
    # Create a channel and stub.
    with grpc.insecure_channel(address) as channel:
        stub = csp_pb2_grpc.CSPServiceStub(channel)

        # Verify the file exists.
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read the file contents in binary mode.
        with open(filepath, "rb") as f:
            data = f.read()

        # Build the request message.
        request = csp_pb2.File(
            filename=os.path.basename(filepath),
            data=data,
        )

        # Send the file to the CSP.
        stub.addData(request)

def main():
    parser = argparse.ArgumentParser(
        description="CSPService client – upload a file."
    )
    parser.add_argument(
        "address",
        help="Network address of the CSP service (e.g., localhost:50051)",
    )
    parser.add_argument(
        "file",
        help="Path to the file to upload",
    )
    args = parser.parse_args()
    run(args.address, args.file)

if __name__ == "__main__":
    main()
