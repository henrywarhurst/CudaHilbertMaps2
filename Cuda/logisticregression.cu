#include <eigen3/Eigen/Core>
#include "logisticregression.cuh"


// Should take in datapoints as eigen matrix (N x 3) input
//const Eigen::VectorXd& runLogisticRegression(const Eigen::MatrixXd &points,
//                                              const Eigen::VectorXi &occupancy)
//{
//	return std::nullptr;	
//}
int getNumBlocks(int numDataPoints)
{
	return (int) ceil((float) numDataPoints / (double) MAX_THREADS);
}

void convertEigenInputToPointers(const Eigen::MatrixXf &points,
                                 const Eigen::MatrixXi &occupancy,
                                 float *h_pointsX,
                                 float *h_pointsY,
                                 float *h_pointsZ,
                                 int   *h_occupancy)
{

}

const Eigen::MatrixXf convertWeightPointerToEigen(float *d_weights, int nWeights)
{
	// Not sure if this will work. May need to provide rows and cols
	// to Map function
	return Eigen::Map<Eigen::MatrixXf>(d_weights, 1, nWeights);
}

__global__ void cudaRbf(float *d_pointsX, float *d_pointsY, float *d_pointsZ,
                        float *d_outputFeatures, int *d_pointIdx,
                        float *d_lengthScale)
{

}

__global__ void cudaSgd(int *d_occupancy,
                        float *d_weights,
                        float *d_features,
                        int *d_pointIdx,
                        curandState_t *states)
{

}

