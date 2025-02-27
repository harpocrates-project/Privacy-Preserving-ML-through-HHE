#!/bin/bash

inotifywait -m /mnt/vol0/dhp/ciphers -e create | while read path events file; do
	python3 analyse.py "10.254.1.10:50052" "$file"
done
