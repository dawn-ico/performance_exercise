#!/bin/bash

# NEED TO source load-env.sh FIRST!

srun -C gpu --partition debug --gres=gpu:1 -u exercise