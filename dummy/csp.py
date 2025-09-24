import os
import grpc
from concurrent import futures

import numpy as np
import tensorflow as tf

import csp_pb2
import csp_pb2_grpc

# Directory to store incoming signal files
DATA_DIR = os.path.abspath("./dummy/data")
os.makedirs(DATA_DIR, exist_ok=True)

# Load model
_model_path = os.path.abspath("./dummy/siesta_1_layer_ml_binary.keras")
if not os.path.isfile(_model_path):
    raise FileNotFoundError(f"Model file not found: {_model_path}")
model = tf.keras.models.load_model(_model_path)
def get_model():
    """Return the pre‑loaded global model."""
    return model


class CSPServiceServicer(csp_pb2_grpc.CSPServiceServicer):
    """Implementation of the CSPService defined in csp.proto."""

    def addData(self, request, context):
        """
        Store the uploaded SpO2 signal text file to disk.

        Args:
            request (csp_pb2.File): protobuf message containing `filename` and raw `data`.
            context: gRPC context (unused).

        Returns:
            csp_pb2.Empty: empty response message.
        """
        filename = os.path.basename(request.filename)
        file_path = os.path.join(DATA_DIR, filename)

        # Write the raw bytes to disk
        with open(file_path, "wb") as fp:
            fp.write(request.data)

        return csp_pb2.Empty()

    def evaluate(self, request, context):
        """
        Evaluate a stored SpO2 signal file using the binary classification model.

        Args:
            request (str): filename of the stored SpO2 signal file (relative to DATA_DIR).
            context: gRPC context (unused).

        Returns:
            csp_pb2.File: protobuf message containing the result file (`filename` and `data`).
        """
        # Resolve the requested file path
        filename = os.path.basename(request.filename)
        file_path = os.path.join(DATA_DIR, filename)

        if not os.path.isfile(file_path):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"File not found: {filename}")
            return csp_pb2.File()

        # Load the SpO2 data file using NumPy for concise parsing
        try:
            data_matrix = np.loadtxt(file_path, delimiter=",")
        except Exception as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Failed to parse data file: {e}")
            return csp_pb2.File()

        # Ensure the data has the proper shape (n_segments, 300)
        if data_matrix.shape[1] != 300:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Each line must contain 300 samples, but found {data_matrix.shape[1]} columns."
            )
            return csp_pb2.File()

        model = get_model()

        # Perform inference
        probs = model.predict(data_matrix, verbose=0)

        # Format output
        preds = np.argmax(probs, axis=1).astype(int)
        pred_str = "\n".join(map(str, preds)) + '\n'
        result_bytes = pred_str.encode("utf-8")
        result_filename = f"{os.path.splitext(filename)[0]}_prediction.txt"

        # Return the result as a File protobuf message
        result_file = csp_pb2.File()
        result_file.filename = result_filename
        result_file.data = result_bytes
        return result_file


def serve(host: str = "[::]", port: int = 50051):
    """Utility to start the gRPC server exposing CSPService."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    csp_pb2_grpc.add_CSPServiceServicer_to_server(CSPServiceServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"CSPService gRPC server listening on port {port}")
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    import grpc

    parser = argparse.ArgumentParser(description="Start CSPService gRPC server.")
    parser.add_argument(
        "address",
        type=str,
        default="0.0.0.0:50052",
        help="Network address to bind the server in the format hostname:port",
    )
    args = parser.parse_args()

    # Parse the address into host and port
    try:
        host, port_str = args.address.rsplit(":", 1)
        port = int(port_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Address must be in the format hostname:port, got '{args.address}'"
        )

    serve(host=host, port=port)
