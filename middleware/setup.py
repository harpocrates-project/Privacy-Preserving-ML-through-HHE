from setuptools import setup, find_packages

setup(
    name='distributed-hhe-ppml-client',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click==8.1.8',
        'grpcio==1.70.0',
        'grpcio-tools==1.70.0',
        'protobuf==5.29.3',
    ],
    entry_points={
        'console_scripts': [
            'distributed-hhe-ppml-client = distributed_hhe_ppml_client:cli',
        ],
    },
)
