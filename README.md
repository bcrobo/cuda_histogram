# Grayscale histogram using cuda

This is an exemple of how cuda can be used to calculate grayscale histogram

## Project Description

The goal of the program is to read images from the data directory. Then two cuda kernels are run on each image.
The first kernel compute grayscale histogram for each block. The second kernel aggregates all the local histogram into the final histogram.
A cpu version of the histogram calculate is provided using opencv.

In the bin directory, the result csv files (named after the image they processed) is saved.
It dumps the cpu and gpu histogram and calculate a per bin difference to ensure the computations are correct.
Then you should obtain 3 lines of 256 bins.

## Code Organization

```bin/```
This folder contains the compile binary along with the resulting csv files (once run).

```data/```
This folder contains 10 imagesfrom the SIPI image database to process. Link of the dataset: https://sipi.usc.edu/database/database.php

```src/```
The source code of the program is only made of one .cu file.

```run.sh```
By running this bash script file you should be able to compile and run the program against the image saved in the data directory.
