#!/bin/bash
set +x
distributed-hhe-ppml-client csp --host "$CSP_HOST" evaluate-model-from-file "$1.bin"

