/*
 * This file is part of CudaHilbertMaps.
 *
 * Copyright (C) 2017 University of Sydney.
 *
 * The use of the code within this file and all code
 * within files that make up the software that is
 * CudaHilbertMaps is permitted for non-commercial 
 * purposes only. The full terms and conditions that
 * apply to the code within this file are detailed 
 * within the LICENSE.txt file unless explicitly 
 * stated. By downloading this file you agree to 
 * comply with these terms.
 *
 * Author: Henry Warhurst
 * 
 */

#ifndef CUDA_LOGISTICREGRESSION_CUH_
#define CUDA_LOGISTICREGRESSION_CUH_

#include <vector>

#include <eigen3/Eigen/Dense>

#include <curand.h>
#include <curand_kernel.h>


// Should take in datapoints as eigen matrix (N x 3) input
Eigen::MatrixXf runLogisticRegression(const Eigen::MatrixXf &points,
									  const Eigen::MatrixXi &occupancy,
									  float lengthScale,
									  float learningRate,
									  float regularisationLambda);

void printStats(size_t nPoints, size_t dataSize, size_t nCudaBlocks);

int getNumBlocks(int numDataPoints, int maxThreads);

void convertEigenInputToPointers(const Eigen::MatrixXf &points,
								 const Eigen::MatrixXi &occupancy,
								 float *h_pointsX,
								 float *h_pointsY,
								 float *h_pointsZ,
								 int   *h_occupancy);

Eigen::MatrixXf convertFloatArrayToEigen(float *h_array, size_t nElements);

std::vector<Eigen::Vector3f> getCloud(	Eigen::MatrixXf weights, 
										Eigen::MatrixXf points, 
										std::vector<std::vector<Eigen::Vector3f> > rays, 
										float lengthScale);

Eigen::MatrixXf getFeatures(Eigen::Vector3f point, const Eigen::MatrixXf &featurePoints, float lengthScale);
								 
__global__ void cudaRbf(float *d_pointsX, 
						float *d_pointsY, 
						float *d_pointsZ,
                        float *d_outputFeatures, 
						float *d_queryX,
						float *d_queryY,
						float *d_queryZ,
                        float *d_lengthScale);

__global__ void cudaRbf(float *d_x,
                        float *d_y,
                        float *d_z,
                        float *d_outputFeatures,
                        float queryX,
                        float queryY,
                        float queryZ,
                        float lengthScale);


__global__ void cudaSgd(int *d_occupancy,
                        float *d_weights,
                        float *d_features,
                        int d_pointIdx,
                        curandState_t *states,
						float learningRate,
						float lambda);

__global__ void initCurand(unsigned int seed, curandState_t* states);


#endif /* CUDA_LOGISTICREGRESSION_CUH_ */
