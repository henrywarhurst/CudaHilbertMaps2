# CudaHilbertMaps
A fast implementation of 3D Hilbert Maps, optimised for operation on an NVIDIA GPU using CUDA. Theoretically should work on any device with compute capability > 2.0, but has thus far only been tested on a device with compute capability 6.1 (__NVIDIA GTX1070__). Put simply, the faster your device, the faster this code will run.

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
