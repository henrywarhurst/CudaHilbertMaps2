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

#include <eigen3/Eigen/Core>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>

#include <curand.h>
#include <curand_kernel.h>

#include "logisticregression.cuh"



/**
 * Access point for computing Hilbert features and running 
 * building logistic regression model using SGD.
 *
 * \param points (N x 3) matrix of (x, y, z) coordinates for each point
 * \param occupancy (N x 1) matrix of {1, -1} occupancy values
 * \param lengthScale length scale to use with RBF kernel
 * \param learningRate SGD learning rate (alpha)
 * \param regularisationLambda regularisation constant (L2 norm)
 *
 * \return (N x 1) matrix of logistic function parameter weights
 *
 */
Eigen::MatrixXf runLogisticRegression(const Eigen::MatrixXf &points,
                                      const Eigen::MatrixXi &occupancy,
                                      float lengthScale,
                                      float learningRate,
                                      float regularisationLambda)
{
	clock_t begin = clock();
	
	// Memory computation
	size_t nPoints = points.rows();
	size_t size = nPoints*sizeof(float);


	// Allocate host memory
	float *h_pointsX, *h_pointsY, *h_pointsZ;
	int *h_occupancy;
	
	h_pointsX 		= (float *) malloc(size);
	h_pointsY 		= (float *) malloc(size);
	h_pointsZ 		= (float *) malloc(size);
	h_occupancy 	= (int *) 	malloc(size);

	// Put input values into raw host memory
	convertEigenInputToPointers(points, occupancy, 
					h_pointsX, h_pointsY, h_pointsZ, h_occupancy);	

	// Allocate device memory
	float *d_pointsX, *d_pointsY, *d_pointsZ;
    int *d_occupancy;
    float *d_weights;
	float *d_features;

    cudaMalloc((void **) &d_pointsX, 	size);
    cudaMalloc((void **) &d_pointsY, 	size); 
	cudaMalloc((void **) &d_pointsZ, 	size);
    cudaMalloc((void **) &d_occupancy, 	size);
    cudaMalloc((void **) &d_weights, 	size);
	cudaMalloc((void **) &d_features, 	size);

	// Copy memory - host to device
	cudaMemcpy(d_pointsX, 	h_pointsX, 		size, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_pointsY, 	h_pointsY, 		size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_pointsZ, 	h_pointsZ, 		size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_occupancy, h_occupancy, 	size, cudaMemcpyHostToDevice);

	// Compute device parameters
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	int maxThreads = props.maxThreadsPerBlock;
	int nBlocks = getNumBlocks(nPoints, maxThreads);	

	printStats(nPoints, size, nBlocks);

	// Setup Random Number Generator
	curandState_t* states;
	cudaMalloc((void **) &states, nBlocks*sizeof(curandState_t));
	initCurand<<<nBlocks, 1>>>(time(0), states);

	std::cout << "Running SGD" << std::endl;
	// Alternate between SGD and computing features
	for (size_t i=0; i<nPoints; ++i) {
		// Compute features for this point
		cudaRbf<<<nBlocks, maxThreads>>>
				(d_pointsX, d_pointsY, d_pointsZ, d_features, i, lengthScale);
		// Run one SGD step using features from this point
		cudaSgd<<<nBlocks, maxThreads>>>
				(d_occupancy, d_weights, d_features, 
								i, states, learningRate, regularisationLambda);
		if (i % 1000 == 0) {
			//std::cout << "." << std::flush;
			std::cout << "\r" << i << "/" << nPoints << " processed";
			std::cout << std::flush;
		}
	}
	std::cout << std::endl;
	std::cout << "SGD complete" << std::endl;

	// Copy weights to host memory
	float *h_weights = (float *) malloc(size);
	cudaMemcpy(h_weights, d_weights, size, cudaMemcpyDeviceToHost);
	Eigen::MatrixXf outputWeights = convertWeightPointerToEigen(h_weights, nPoints); 
	

	// Delete device memory
	cudaFree(d_pointsX);
	cudaFree(d_pointsY); cudaFree(d_pointsZ); cudaFree(d_occupancy);
	cudaFree(d_occupancy);
	cudaFree(d_weights);
	cudaFree(d_features);

	// Delete host memory
	free(h_pointsX);
	free(h_pointsY);
	free(h_pointsZ);
	free(h_occupancy);
	
	clock_t end = clock();
	double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << "Took " << elapsedSecs << " seconds to train" << std::endl;

	return outputWeights;	
}

/**
 * \brief Print out information about GPU memory
 *
 * \param nPoints the number of points in the dataset
 * \param dataSize the amount of bytes occupied by the points data
 * \param nCudaBlocks the number of CUDA blocks to use
 */
void printStats(size_t nPoints, size_t dataSize, size_t nCudaBlocks)
{
	std::cout << "********* CUDA LOGISTIC REGRESSION *********" << std::endl;
	std::cout << "Number of points: " << nPoints << std::endl;
	std::cout << "Amount of memory for points: ";
	std::cout << dataSize/1000000.0 << "MB" << std::endl;
	std::cout << "Using " << nCudaBlocks << " blocks of memory on the GPU";
	std::cout << std::endl;
}


int getNumBlocks(int numDataPoints, int maxThreads)
{
	return (int) ceil((float) numDataPoints / (double) maxThreads);
}

/**
 * \brief The caller of this function is responsible for allocating 
 * memory for h_pointsX, h_pointsY, h_pointsZ, and h_occupancy.
 * Each of these variables needs sizeof(float) * N where N is 
 * number of points.
 *
 * \param points (N x 3) matrix of points, (x,y,z) order for each row.
 * \param occupancy (N x 1) matrix of occupancies (-1 is free space, 1 is occupied)
 * \param h_pointsX raw C output storage of X values
 * \param h_pointsY raw C output storage of Y values
 * \param h_pointsZ raw C output storage of Z values
 * \param h_occupancy raw C output storage of occupancy values
 */
void convertEigenInputToPointers(const Eigen::MatrixXf &points,
                                 const Eigen::MatrixXi &occupancy,
                                 float *h_pointsX,
                                 float *h_pointsY,
                                 float *h_pointsZ,
                                 int   *h_occupancy)
{
	// TODO: Add some kind of size checking here
	float *fullPointsOutput = (float *) malloc(sizeof(float) * points.size());
	int *occupancyOutput = (int *) malloc(sizeof(int) * occupancy.rows());

	// Convert eigen matrices to raw C data types
	Eigen::Map<Eigen::MatrixXf>(fullPointsOutput, points.rows(), points.cols()) = 
						points;
	Eigen::Map<Eigen::MatrixXi>(occupancyOutput, occupancy.rows(), occupancy.cols()) =
						occupancy;

	// Copy data 
	std::copy(fullPointsOutput + 0 * points.rows(), 
			  fullPointsOutput + 1 * points.rows(), 
			  h_pointsX);
	std::copy(fullPointsOutput + 1 * points.rows(), 
			  fullPointsOutput + 2 * points.rows(), 
			  h_pointsY);
	std::copy(fullPointsOutput + 2 * points.rows(), 
			  fullPointsOutput + 3 * points.rows(), 
			  h_pointsZ);

	std::copy(occupancyOutput, 
			  occupancyOutput + occupancy.rows(), 
			  h_occupancy);	
	
}

/**
 * \brief Convert raw sigmoid function weights to Eigen matrix
 * 
 * \param h_weights the raw weights generated by SGD optimisation
 * \param nWeights the length of h_weights
 *
 * \return The weights as a column vector (N x 1 matrix)
 */
Eigen::MatrixXf convertWeightPointerToEigen(float *h_weights, size_t nWeights)
{
	return Eigen::Map<Eigen::MatrixXf>(h_weights, 1, nWeights);
}

/**
 * \brief Computes RBF kernel using CUDA
 * 
 * \param d_x device memory storing x coordinates
 * \param d_y device memory storing y coordinates
 * \param d_z device memory storing z coordinates
 * \param outputFeatures device memory to store computed feature vector
 * \param d_pointIdx the index of the point in the dataset that we are looking at
 * \param d_lengthScale the length scale to use when computing the RBF function
 */
__global__ void cudaRbf(float *d_x, float *d_y, float *d_z,
                        float *outputFeatures, int d_pointIdx,
                        float d_lengthScale)
{
    int cudaIdx = threadIdx.x + blockIdx.x * blockDim.x;

    float diff = (d_x[cudaIdx] - d_x[d_pointIdx]) * (d_x[cudaIdx] - d_x[d_pointIdx]) +
                 (d_y[cudaIdx] - d_y[d_pointIdx]) * (d_y[cudaIdx] - d_y[d_pointIdx]) +
                 (d_z[cudaIdx] - d_z[d_pointIdx]) * (d_z[cudaIdx] - d_z[d_pointIdx]);

	outputFeatures[cudaIdx] = (float) exp(-d_lengthScale * diff);
}

/**
 * \brief Perform weight update SGD step
 * 
 * \param d_occupancy device memory storing output value for each point in dataset
 * \param d_weights the logistic function weights that are being updated
 * \param d_features the features for the given point as computed by cudaRbf
 * \param d_pointIdx the index of the current point in the dataset
 * \param states initialiser for the random number generator
 * \param learningRate the alpa value to use with SGD
 * \param lambda the regularisation constant for L2 regularisation
 */
__global__ void cudaSgd(int *d_occupancy,
                        float *d_weights,
                        float *d_features,
                        int d_pointIdx,
                        curandState_t *states,
						float learningRate,
						float lambda)
{
    int cudaIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // If this is is the first example, just initialise the weights
	if (d_pointIdx == 0) {
	      // Random value between 0 and 1
	      d_weights[cudaIdx] = (float) (curand(&states[blockIdx.x]) % 1000) / 1000.0; 
	} else {
		float numerator 	= -d_occupancy[d_pointIdx] * d_features[cudaIdx];
    	float denominator 	= 1 + exp(-numerator*d_weights[cudaIdx]);         

        // Just using L2 regularisation here, may use elastic net later
        float regulariser = lambda*d_weights[cudaIdx];

        // Combine all the parts
        float lossGradient = (numerator/denominator) + regulariser;

        // Update weight
        d_weights[cudaIdx] = d_weights[cudaIdx] - learningRate*lossGradient;
	}
}

/**
 * \brief Initialises the random number generators
 * 
 * \param seed seed for the random number generator
 * \param curandState_t the array of generators to initisalise
 */
__global__ void initCurand(unsigned int seed, curandState_t* states)
{
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}
