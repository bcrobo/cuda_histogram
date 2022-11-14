#!/bin/bash

BUILD_DIR=build
BIN_DIR=bin

if [ ! -d "$BUILD_DIR" ]; then
  mkdir build
fi

if [ ! -d "$BIN_DIR" ]; then
  mkdir bin
fi

cd bin/
rm -r *
cd ../build
rm -r *
cmake ../ -DCMAKE_INSTALL_PREFIX=../bin
make -j4
make install
cd ../bin
./cuda_histogram ../data/
