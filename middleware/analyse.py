#!/usr/bin/env python3

from csp import CSPService

if __name__ == '__main__':
    import sys
    with CSPService(sys.argv[1]) as csp_service:
        print('Evaluating model')
        csp_service.evaluate_model_from_file(sys.argv[2])
