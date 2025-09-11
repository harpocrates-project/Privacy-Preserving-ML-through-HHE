# Build Stage
FROM debian:bookworm-slim AS build

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        cmake \
        git
WORKDIR /distributed_hhe_ppml
COPY . ./
RUN cmake -S . -B build -DCMAKE_PREFIX_PATH=libs/seal && cmake --build build

# Install Stage
FROM debian:bookworm

COPY --from=build /distributed_hhe_ppml/build/analyst \
    /distributed_hhe_ppml/build/csp \
    /distributed_hhe_ppml/build/user \
    /usr/local/bin/

WORKDIR /srv
