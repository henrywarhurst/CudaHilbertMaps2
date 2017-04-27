#include <eigen3/Eigen/Core>
#include <cmath>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

#include "logisticregression.cuh"


// Should take in datapoints as eigen matrix (N x 3) input
Eigen::MatrixXf runLogisticRegression(const Eigen::MatrixXf &points,
                                      const Eigen::MatrixXi &occupancy,
                                      float lengthScale,
                                      float learningRate,
                                      float regularisationLambda)
{
	// Memory computation
	size_t nPoints = points.size();
	size_t size = nPoints*sizeof(float);
	std::cout << "Number of points: " << nPoints << std::endl;
	std::cout << "Amount of memory for points" << std::endl;
	std::cout << size << std::endl;

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

	std::cout << "Got past the eigen conversion" << std::endl;

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
	int numBlocks = getNumBlocks(nPoints, maxThreads);	
	std::cout << "Using " << numBlocks << " blocks of memory on the GPU";
	std::cout << std::endl;

	// Setup Random Number Generator
	curandState_t* states;
	cudaMalloc((void **) &states, numBlocks*sizeof(curandState_t));
	initCurand<<<numBlocks, 1>>>(time(0), states);

	// Alternate between SGD and computing features
	for (size_t i=0; i<nPoints; ++i) {
		// Compute features for this point
		cudaRbf<<<numBlocks, maxThreads>>>
				(d_pointsX, d_pointsY, d_pointsZ, d_features, i, lengthScale);
		// Run one SGD step using features from this point
		cudaSgd<<<numBlocks, maxThreads>>>
				(d_occupancy, d_weights, d_features, 
								i, states, learningRate, regularisationLambda);
	}

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

	return outputWeights;	
}

int getNumBlocks(int numDataPoints, int maxThreads)
{
	return (int) ceil((float) numDataPoints / (double) maxThreads);
}

/**
 * The caller of this function is responsible for allocating 
 * memory for h_pointsX, h_pointsY, h_pointsZ, and h_occupancy.
 * Each of these variables needs sizeof(float) * N where N is 
 * number of points.
 *
 * @param points (N x 3) matrix of points, (x,y,z) order for each row.
 * @param occupancy (N x 1) matrix of occupancies (-1 is free space, 1 is occupied)
 * @param h_pointsX raw C output storage of X values
 * @param h_pointsY raw C output storage of Y values
 * @param h_pointsZ raw C output storage of Z values
 * @param h_occupancy raw C output storage of occupancy values
 */
void convertEigenInputToPointers(const Eigen::MatrixXf &points,
                                 const Eigen::MatrixXi &occupancy,
                                 float *h_pointsX,
                                 float *h_pointsY,
                                 float *h_pointsZ,
                                 int   *h_occupancy)
{
	// TODO: Add some kind of size checking here
	float *fullPointsOutput = (float *) malloc(sizeof(float) * points.rows() * points.cols());
	int *occupancyOutput = (int *) malloc(sizeof(int) * occupancy.cols());

	// Coordinate copy
	std::copy(fullPointsOutput + 0 * points.rows(), 
			  fullPointsOutput + 1 * points.rows(), 
			  h_pointsX);
	std::copy(fullPointsOutput + 1 * points.rows(), 
			  fullPointsOutput + 2 * points.rows(), 
			  h_pointsY);
	std::copy(fullPointsOutput + 2 * points.rows(), 
			  fullPointsOutput + 3 * points.rows(), 
			  h_pointsZ);

	// Occupancy copy
	std::copy(occupancyOutput, 
			  occupancyOutput + occupancy.rows(), 
			  h_occupancy);	
	
}

Eigen::MatrixXf convertWeightPointerToEigen(float *h_weights, size_t nWeights)
{
	return Eigen::Map<Eigen::MatrixXf>(h_weights, 1, nWeights);
}


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

__global__ void initCurand(unsigned int seed, curandState_t* states)
{
    curand_init(seed, blockIdx.x, 0, &states[blockIdx.x]);
}
