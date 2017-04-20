#ifndef CUDA_LOGISTICREGRESSION_CUH_
#define CUDA_LOGISTICREGRESSION_CUH_

#include <eigen3/Eigen/Dense>

#include <curand.h>
#include <curand_kernel.h>

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

// Should take in datapoints as eigen matrix (N x 3) input
const Eigen::VectorXd& runLogisticRegression(const Eigen::MatrixXd &points,
											  const Eigen::VectorXi &occupancy);

int getNumBlocks(int numDataPoints);

void convertEigenInputToPointers(const Eigen::MatrixXd &points,
								 const Eigen::VectorXi &occupancy,
								 float *h_pointsX,
								 float *h_pointsY,
								 float *h_pointsZ,
								 int   *h_occupancy);

const Eigen::VectorXd& convertWeightPointerToEigen(float *d_weights);
								 
__global__ void cudaRbf(float *d_pointsX, float *d_pointsY, float *d_pointsZ,
                        float *d_outputFeatures, int *d_pointIdx,
                        float *d_lengthScale);

__global__ void cudaSgd(int *d_occupancy,
                        float *d_weights,
                        float *d_features,
                        int *d_pointIdx,
                        curandState_t *states);


#endif /* CUDA_LOGISTICREGRESSION_CUH_ */
