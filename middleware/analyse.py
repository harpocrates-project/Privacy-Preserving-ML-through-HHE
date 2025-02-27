#!/usr/bin/env python3

import base64
import grpc

from hhe_pb2_grpc import CSPServiceStub
from hhe_pb2 import DataFile, EncSymmetricDataMsg


class CSPService:
    def __init__(self, host):
        self._channel = grpc.insecure_channel(host)
        self._stub = CSPServiceStub(self._channel)

    def evaluate_model_from_file(self, filename):
        data_file = DataFile(filename=filename)
        return self._stub.evaluateModelFromFile(data_file)

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self
              
    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.close()


if __name__ == '__main__':
    import sys
    with CSPService(sys.argv[1]) as csp_service:
        print('Evaluating model')
        csp_service.evaluate_model_from_file(sys.argv[2])
