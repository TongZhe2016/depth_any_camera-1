#!/usr/bin/env bash
conda activate dac && \
export PYTHONPATH="$PWD:$PYTHONPATH" && \
cd dac/models/ops/ && \
pip install -e . && \
cd  ../../../