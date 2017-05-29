# CudaHilbertMaps
A fast implementation of 3D Hilbert Maps, optimised for operation on an NVIDIA GPU using CUDA. Theoretically should work on any device with compute capability > 2.0, but has thus far only been tested on a device with compute capability 6.1 (__NVIDIA GTX1070__). Put simply, the faster your device, the faster this code will run.

## What is a Hilbert Map?

Please see the [original research publication](http://www-personal.acfr.usyd.edu.au/f.ramos/Fabio_Ramos_Homepage/Publications_files/hilbertmaps_rss2015.pdf), as presented at RSS 2015 

## Dependencies

* CUDA
* [PCL](http://pointclouds.org/)
* [Eigen](https://github.com/stevenlovegrove/eigen)

## Building

```bash
mkdir build
cd build
cmake ..
make -j
```

## Running

```bash
./output
```

## Obtaining .occ files for input to this code

The input files (.occ files) are generated from [HilbertMapICP](https://github.com/henrywarhurst/HilbertMapICP/tree/feature/rgb-support).
