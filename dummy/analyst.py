import argparse
import os
import sys

import grpc

# Import the generated classes
import csp_pb2
import csp_pb2_grpc

# Directory to store incoming signal files
DATA_DIR = os.path.abspath("./dummy/data")
os.makedirs(DATA_DIR, exist_ok=True)


class Analyst:
    """Client for the CSPService."""

    def __init__(self, address: str):
        """Create a channel and stub for the given address."""
        self.channel = grpc.insecure_channel(address)
        self.stub = csp_pb2_grpc.CSPServiceStub(self.channel)

    def evaluate(self, filename: str) -> None:
        """Call evaluate RPC and write the returned file to disk."""
        request = csp_pb2.StoredFile(filename=filename)
        try:
            response = self.stub.evaluate(request)
        except grpc.RpcError as e:
            sys.stderr.write(f"gRPC error while calling evaluate: {e}\\n")
            sys.exit(1)

        # Determine output filename
        base_name = os.path.basename(filename).removesuffix('_data.txt')
        output_name = f"{DATA_DIR}/{base_name}_plaintext_binaryoutput.txt"
        try:
            with open(output_name, "wb") as out_file:
                out_file.write(response.data)

            print(f"Plaintext output written to: {output_name}")
        except OSError as e:
            sys.stderr.write(f"Failed to write output file: {e}\\n")
            sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyst client for CSPService evaluate RPC"
    )
    parser.add_argument(
        "address",
        type=str,
        help="Network address of the CSPService (e.g., localhost:50051)",
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Name of the file to send for evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = Analyst(args.address)
    client.evaluate(args.filename)


if __name__ == "__main__":
    main()
