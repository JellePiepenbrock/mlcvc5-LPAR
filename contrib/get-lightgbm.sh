#!/bin/bash
#
# File:  get-lightgbm.sh
# Author:  mikolas
# Created on:  Fri May 14 08:16:31 WEST 2021
# Copyright (C) 2021, Mikolas Janota
#
mkdir -p deps/install/include/
mkdir -p deps/install/lib
cd deps/install/include/
wget sat.inesc-id.pt/~mikolas/lightgbm.h
cd -
cd deps/install/lib/
wget sat.inesc-id.pt/~mikolas/liblightgbm.a
cd -
