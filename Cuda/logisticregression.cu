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
//#include <cmath>
#include <iostream>
#include <string>
#include <ctime>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>

#include "logisticregression.cuh"



/**
 * \brief Access point for computing Hilbert features and running 
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
	
	// memory computation
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
	float *d_queryX, *d_queryY, *d_queryZ;
    int *d_occupancy;
    float *d_weights;
	float *d_features;
	float *d_lengthScale;

    cudaMalloc((void **) &d_pointsX, 	size);
    cudaMalloc((void **) &d_pointsY, 	size); 
	cudaMalloc((void **) &d_pointsZ, 	size);
	cudaMalloc((void **) &d_queryX, 	sizeof(float));
	cudaMalloc((void **) &d_queryY, 	sizeof(float));
	cudaMalloc((void **) &d_queryZ, 	sizeof(float));
    cudaMalloc((void **) &d_occupancy, 	size);
    cudaMalloc((void **) &d_weights, 	size);
	cudaMalloc((void **) &d_features, 	size);
	cudaMalloc((void **) &d_lengthScale,sizeof(float));

	// Copy memory - host to device
	cudaMemcpy(d_pointsX, 	h_pointsX, 		size, cudaMemcpyHostToDevice);
 	cudaMemcpy(d_pointsY, 	h_pointsY, 		size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_pointsZ, 	h_pointsZ, 		size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_occupancy, h_occupancy, 	size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lengthScale, &lengthScale, sizeof(float), cudaMemcpyHostToDevice);

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

	std::cout << "The number of threads per block is: " << maxThreads << std::endl;

	std::cout << "Running SGD" << std::endl;
	// Alternate between SGD and computing features
	for (size_t i=0; i<nPoints; ++i) {
		// Copy current point to the device
		cudaMemcpy(d_queryX, &h_pointsX[i], sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_queryY, &h_pointsY[i], sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_queryZ, &h_pointsZ[i], sizeof(float), cudaMemcpyHostToDevice);

		// Compute features for this point
		cudaRbf<<<nBlocks, maxThreads>>>(	d_pointsX, 
											d_pointsY, 
											d_pointsZ, 
											d_features, 
											d_queryX, 
											d_queryY, 
											d_queryZ, 
											d_lengthScale);
		// Run one SGD step using features extracted from current point
		cudaSgd<<<nBlocks, maxThreads>>>(	d_occupancy, 
											d_weights, 
											d_features, 
											i, 
											states, 
											learningRate, 
											regularisationLambda);
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
	Eigen::MatrixXf outputWeights = convertFloatArrayToEigen(h_weights, nPoints); 
	

	// Delete device memory
	cudaFree(d_pointsX);
	cudaFree(d_pointsY); 
	cudaFree(d_pointsZ); 
	cudaFree(d_occupancy);
	cudaFree(d_occupancy);
	cudaFree(d_weights);
	cudaFree(d_features);
	cudaFree(d_queryX);
	cudaFree(d_queryY);
	cudaFree(d_queryZ);
	cudaFree(d_lengthScale);

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

void runLinearRegression(std::vector<float> x,
						 std::vector<float> y,
						 std::vector<float> z,
						 std::vector<int> r,
						 std::vector<int> g,
						 std::vector<int> b,
						 float lengthScale,
						 float learningRate,
						 float regularisationLambda)
{
	// Convert vectors to raw arrays and copy to device
	size_t numPoints = x.size();	
	size_t pointStorage = numPoints * sizeof(float);
	size_t colourStorage = numPoints * sizeof(int);

	float* d_x;
	float* d_y;
	float* d_z;

	int* d_r;
	int* d_g;
	int* d_b;

	cudaMalloc((void **) &d_x, pointStorage);
	cudaMalloc((void **) &d_y, pointStorage);
	cudaMalloc((void **) &d_z, pointStorage);

	cudaMalloc((void **) &d_r, colourStorage);
	cudaMalloc((void **) &d_g, colourStorage);
	cudaMalloc((void **) &d_b, colourStorage);

	cudaMemcpy(d_x, x.data(), pointStorage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), pointStorage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, z.data(), pointStorage, cudaMemcpyHostToDevice);

	cudaMemcpy(d_r, r.data(), colourStorage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_g, g.data(), colourStorage, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), colourStorage, cudaMemcpyHostToDevice);	


	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);

	cudaFree(d_r);
	cudaFree(d_g);
	cudaFree(d_b);
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
Eigen::MatrixXf convertFloatArrayToEigen(float *h_array, size_t nElements)
{
	return Eigen::Map<Eigen::MatrixXf>(h_array, 1, nElements);
}


std::vector<Eigen::Vector3f> getCloud(	Eigen::MatrixXf weights, 
										Eigen::MatrixXf points, 
										std::vector<std::vector<Eigen::Vector3f> > rays, 
										float lengthScale)
{
	size_t rayLength = rays[0].size();
	std::cout << "Raylength = " << rayLength << std::endl;
	size_t nRays = rays.size();
	std::cout << "Number of rays = " << nRays << std::endl;
	size_t nPoints = rayLength * nRays;
	size_t querySize = sizeof(float) * nPoints;

	float *queryX, *queryY, *queryZ;

	queryX = (float *) malloc(querySize);
	queryY = (float *) malloc(querySize);
	queryZ = (float *) malloc(querySize);

	// Copy rays into raw C arrays
	for (size_t i=0; i<nRays; ++i) {
		std::vector<Eigen::Vector3f> curRay = rays[i];
		for (size_t j=0; j<rayLength; ++j) {
			size_t curIdx = i*rayLength + j;
			queryX[curIdx] = curRay[j](0);
			queryY[curIdx] = curRay[j](1);
			queryZ[curIdx] = curRay[j](2);	
		}
	}

	float *pointsX, *pointsY, *pointsZ;
	size_t nInducingPoints = points.rows();
	size_t inducingPointsSize = points.rows() * sizeof(float);

	pointsX = (float *) malloc(inducingPointsSize);
	pointsY = (float *) malloc(inducingPointsSize);
	pointsZ = (float *) malloc(inducingPointsSize);


    // Copy inducing points data to raw c arrays
    float *fullPointsOutput = (float *) malloc(sizeof(float) * points.size());

    Eigen::Map<Eigen::MatrixXf>(fullPointsOutput, points.rows(), points.cols()) =
                            points;

    std::copy(fullPointsOutput + 0 * points.rows(),
              fullPointsOutput + 1 * points.rows(),
              pointsX);
    std::copy(fullPointsOutput + 1 * points.rows(),
              fullPointsOutput + 2 * points.rows(),
              pointsY);
    std::copy(fullPointsOutput + 2 * points.rows(),
              fullPointsOutput + 3 * points.rows(),
              pointsZ);

	// Copy weights data into raw c array
	float *h_weights = (float *) malloc(sizeof(float) * weights.size());
	Eigen::Map<Eigen::MatrixXf>(h_weights, weights.rows(), weights.cols()) = weights;


	// Copy raw C arrays to device
	float *d_pointsX, *d_pointsY, *d_pointsZ;
	float *d_queryX, *d_queryY, *d_queryZ;
	float *d_features;
	float *d_lengthScale;
	float *d_weights;
	
	cudaMalloc((void **) &d_queryX, querySize);
	cudaMalloc((void **) &d_queryY, querySize);
	cudaMalloc((void **) &d_queryZ, querySize);
	cudaMalloc((void **) &d_pointsX, inducingPointsSize);
	cudaMalloc((void **) &d_pointsY, inducingPointsSize);
	cudaMalloc((void **) &d_pointsZ, inducingPointsSize); 
	cudaMalloc((void **) &d_weights, inducingPointsSize);
	cudaMalloc((void **) &d_features, inducingPointsSize);
	cudaMalloc((void **) &d_lengthScale, sizeof(float));

	cudaMemcpy(d_queryX, queryX, 	querySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_queryY, queryY, 	querySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_queryZ, queryZ, 	querySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pointsX, pointsX, 	inducingPointsSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pointsY, pointsY, 	inducingPointsSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pointsZ, pointsZ, 	inducingPointsSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, h_weights, inducingPointsSize, cudaMemcpyHostToDevice);	
	cudaMemcpy(d_lengthScale, &lengthScale, querySize, cudaMemcpyHostToDevice);

    // Compute device parameters
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    int maxThreads = props.maxThreadsPerBlock;
    int nBlocks = getNumBlocks(nPoints, maxThreads);

	thrust::device_ptr<float> d_ptr_weights = thrust::device_pointer_cast(d_weights);

	std::vector<Eigen::Vector3f> cloud;

	for (size_t i=0; i<nPoints; ++i) {
		std::cout << "\r" << i << "/" << nPoints << " points processed" << std::flush;

		// Invoke CUDA kernel
	    cudaRbf<<<nBlocks, maxThreads>>>(   d_pointsX,
	                                        d_pointsY,
	                                        d_pointsZ,
	                                        d_features,
	                                        queryX[i],
	                                        queryY[i],
	                                        queryZ[i],
	                                        lengthScale);

		// Dot product features with weights
		thrust::device_ptr<float> d_ptr_features = thrust::device_pointer_cast(d_features);
		float dotResult = thrust::inner_product(d_ptr_weights, d_ptr_weights + nInducingPoints, d_ptr_features, 0.0);
		float logitResult = 1/(1 + exp(-dotResult));

		if (logitResult > 0.5) {
			std::cout << "Greater than 0.5" << std::endl;
			Eigen::Vector3f newCloudPoint;
			newCloudPoint << queryX[i], queryY[i], queryZ[i];
			cloud.push_back(newCloudPoint);
			
			size_t tmp = i / rayLength;
			i = (tmp + 1)*rayLength;
		}	
	}


	free(queryX);
	free(queryY);
	free(queryZ);
	free(pointsX);
	free(pointsY);
	free(pointsZ);
	free(fullPointsOutput);
	free(h_weights);

	cudaFree(d_queryX);
	cudaFree(d_queryY);
	cudaFree(d_queryZ);
	cudaFree(d_pointsX);
	cudaFree(d_pointsY);
	cudaFree(d_pointsZ);
	cudaFree(d_weights);
	cudaFree(d_features);
	cudaFree(d_lengthScale);

	return cloud;
}


// TODO: COMMENT THIS FUNCTION
Eigen::MatrixXf getFeatures(Eigen::Vector3f point, const Eigen::MatrixXf &featurePoints, float lengthScale)
{		
	// memory computation
	size_t nPoints = featurePoints.rows();
	size_t size = nPoints*sizeof(float);

	// Compute device parameters
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	int maxThreads = props.maxThreadsPerBlock;
	int nBlocks = getNumBlocks(nPoints, maxThreads);	

	// Allocate host memory
	float *h_pointsX, *h_pointsY, *h_pointsZ;
	float h_queryX, h_queryY, h_queryZ;
	
	h_pointsX 		= (float *) malloc(size);
	h_pointsY 		= (float *) malloc(size);
	h_pointsZ 		= (float *) malloc(size);

	// Extract input point values
	h_queryX = point(0);
	h_queryY = point(1);
	h_queryZ = point(2);

	// Copy points data to raw c arrays
	float *fullPointsOutput = (float *) malloc(sizeof(float) * featurePoints.size());
	
	Eigen::Map<Eigen::MatrixXf>(fullPointsOutput, featurePoints.rows(), featurePoints.cols()) = 
							featurePoints;

	std::copy(fullPointsOutput + 0 * featurePoints.rows(), 
			  fullPointsOutput + 1 * featurePoints.rows(), 
			  h_pointsX);
	std::copy(fullPointsOutput + 1 * featurePoints.rows(), 
			  fullPointsOutput + 2 * featurePoints.rows(), 
			  h_pointsY);
	std::copy(fullPointsOutput + 2 * featurePoints.rows(), 
			  fullPointsOutput + 3 * featurePoints.rows(), 
			  h_pointsZ);

	// Allocate device memory
	float *d_pointsX, *d_pointsY, *d_pointsZ;
	float *d_queryX, *d_queryY, *d_queryZ;
	float *d_features;
	float *d_lengthScale;

    cudaMalloc((void **) &d_pointsX, 	size);
    cudaMalloc((void **) &d_pointsY, 	size); 
	cudaMalloc((void **) &d_pointsZ, 	size);
	cudaMalloc((void **) &d_features,	size);
	cudaMalloc((void **) &d_queryX, 	sizeof(float));
	cudaMalloc((void **) &d_queryY, 	sizeof(float));
	cudaMalloc((void **) &d_queryZ, 	sizeof(float));
	cudaMalloc((void **) &d_lengthScale,sizeof(float));
    
	cudaMemcpy(d_pointsX, 		h_pointsX, 		size, 			cudaMemcpyHostToDevice);
 	cudaMemcpy(d_pointsY, 		h_pointsY, 		size, 			cudaMemcpyHostToDevice); 
	cudaMemcpy(d_pointsZ, 		h_pointsZ, 		size, 			cudaMemcpyHostToDevice);	
	cudaMemcpy(d_lengthScale, 	&lengthScale, 	sizeof(float), 	cudaMemcpyHostToDevice);
	cudaMemcpy(d_queryX,		&h_queryX,		sizeof(float),	cudaMemcpyHostToDevice);
	cudaMemcpy(d_queryY, 		&h_queryY, 		sizeof(float), 	cudaMemcpyHostToDevice);		
	cudaMemcpy(d_queryZ, 		&h_queryZ, 		sizeof(float), 	cudaMemcpyHostToDevice);

	// Cuda Kernel
	cudaRbf<<<nBlocks, maxThreads>>>(	d_pointsX, 
										d_pointsY, 
										d_pointsZ, 
										d_features, 
										d_queryX, 
										d_queryY, 
										d_queryZ, 
										d_lengthScale);

	float *h_features;
	h_features = (float *) malloc(size);
	cudaMemcpy(h_features, d_features, size, cudaMemcpyDeviceToHost);

	Eigen::MatrixXf vectorFeatures = convertFloatArrayToEigen(h_features, nPoints);

	free(h_pointsX);
	free(h_pointsY);
	free(h_pointsZ);
	free(h_features);
	free(fullPointsOutput);

	cudaFree(d_pointsX);
	cudaFree(d_pointsY);
	cudaFree(d_pointsZ);
	cudaFree(d_features);
	cudaFree(d_queryX);
	cudaFree(d_queryY);
	cudaFree(d_queryZ);
	cudaFree(d_lengthScale);


	return vectorFeatures;
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
__global__ void cudaRbf(float *d_x, 
						float *d_y, 
						float *d_z,
                        float *d_outputFeatures, 
						float *d_queryX,
						float *d_queryY,
						float *d_queryZ,
                        float *d_lengthScale)
{
    int cudaIdx = threadIdx.x + blockIdx.x * blockDim.x;

    float diff = (d_x[cudaIdx] - *d_queryX) * (d_x[cudaIdx] - *d_queryX) +
                 (d_y[cudaIdx] - *d_queryY) * (d_y[cudaIdx] - *d_queryY) +
                 (d_z[cudaIdx] - *d_queryZ) * (d_z[cudaIdx] - *d_queryZ);

	d_outputFeatures[cudaIdx] = (float) exp(-*d_lengthScale * diff);
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
__global__ void cudaRbf(float *d_x,
                        float *d_y,
                        float *d_z,
                        float *d_outputFeatures,
                        float queryX,
                        float queryY,
                        float queryZ,
                        float lengthScale)
{
    int cudaIdx = threadIdx.x + blockIdx.x * blockDim.x;

    float diff = (d_x[cudaIdx] - queryX) * (d_x[cudaIdx] - queryX) +
                 (d_y[cudaIdx] - queryY) * (d_y[cudaIdx] - queryY) +
                 (d_z[cudaIdx] - queryZ) * (d_z[cudaIdx] - queryZ);

    d_outputFeatures[cudaIdx] = (float) exp(-lengthScale * diff);
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
	if (d_pointIdx == 0) {
		d_weights[cudaIdx] = 0;
	}

	float precomp 	= d_occupancy[d_pointIdx] * d_features[cudaIdx];
	float lossGradient = (-precomp)/(1+exp(precomp*d_weights[cudaIdx])) + lambda*d_weights[cudaIdx]*d_weights[cudaIdx];

	// Update weight
	d_weights[cudaIdx] = d_weights[cudaIdx] - learningRate*lossGradient;
}

__global__ void cudaLinearRegressionSgd(int *d_colourChannel,
										float *d_weights,
										float *d_features,
										int d_pointIdx,
										float learningRate,
										float lambda)
{
	int cudaIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (d_pointIdx == 0) {
		d_weights[cudaIdx] = 0;
	}

	// Compute the gradient
	float diff = d_colourChannel[d_pointIdx] - d_features[cudaIdx]*d_weights[cudaIdx];
	float lossGradient = diff * d_features[cudaIdx];

	// Update the weight
	d_weights[cudaIdx] = d_weights[cudaIdx] - learningRate*lossGradient;
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
