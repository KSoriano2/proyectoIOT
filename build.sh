#!/usr/bin/env bash

apt-get update && apt-get install -y \
  build-essential \
  cmake \
  libboost-all-dev \
  libopenblas-dev \
  liblapack-dev \
  libx11-dev \
  libgtk-3-dev \
  python3-dev

pip install -r requirements.txt
