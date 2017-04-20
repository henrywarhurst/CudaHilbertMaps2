#ifndef CUDA_LOGISTICREGRESSION_CUH_
#define CUDA_LOGISTICREGRESSION_CUH_

#include <Eigen/Dense>

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

// Should take in datapoints as eigen matrix (N x 3) input
const Eigen::VectorXd& runLogisiticRegression(const Eigen::MatrixXd &points,
											  const Eigen::VectorXi &occupancy);

void initSgdParams(float learningRate, float lambda);

void initRbfParams(float lengthScale);

void getCudaParams(int numDataPoints);

void convertEigenInputToPointers(const Eigen::MatrixXd &points,
								 const Eigen::VectorXi &occupancy
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
