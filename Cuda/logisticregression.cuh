#ifndef CUDA_LOGISTICREGRESSION_CUH_
#define CUDA_LOGISTICREGRESSION_CUH_

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

Eigen::MatrixXf convertWeightPointerToEigen(float *h_weights, size_t numPoints);
								 
__global__ void cudaRbf(float *d_pointsX, float *d_pointsY, float *d_pointsZ,
                        float *d_outputFeatures, int d_pointIdx,
                        float d_lengthScale);

__global__ void cudaSgd(int *d_occupancy,
                        float *d_weights,
                        float *d_features,
                        int d_pointIdx,
                        curandState_t *states,
						float learningRate,
						float lambda);

__global__ void initCurand(unsigned int seed, curandState_t* states);


#endif /* CUDA_LOGISTICREGRESSION_CUH_ */
