#!/usr/bin/env python3

import base64
import grpc

from .hhe_pb2_grpc import CSPServiceStub
from .hhe_pb2 import DataFile, EncSymmetricDataMsg, EncSymmetricDataRecord


class CSPService:
    def __init__(self, host=None):
        self._channel = grpc.insecure_channel(host)
        self._stub = CSPServiceStub(self._channel)

    def evaluate_model_from_file(self, filename):
        return self._stub.evaluateModelFromFile(
            DataFile(filename=filename)
        )

    def add_encrypted_data(self, record, patient_id):
        return self._stub.addEncryptedData(
            EncSymmetricDataMsg(
                record=[EncSymmetricDataRecord(value=[r]) for r in record],
                patientID=patient_id
            )
        )

    def close(self):
        self._channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        self.close()


# Analyst Service
# TODO
# User Service 
# TODO