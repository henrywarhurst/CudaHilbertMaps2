#include <eigen3/Eigen/Core>
#include "logisticregression.cuh"


 Should take in datapoints as eigen matrix (N x 3) input
Eigen::MatrixXf runLogisticRegression(const Eigen::MatrixXd &points,
                                              const Eigen::VectorXi &occupancy)
{
		
}

int getNumBlocks(int numDataPoints)
{
	return (int) ceil((float) numDataPoints / (double) MAX_THREADS);
}

// Make sure the raw c pointers are prealloc'd
void convertEigenInputToPointers(const Eigen::MatrixXf &points,
                                 const Eigen::MatrixXi &occupancy,
                                 float *h_pointsX,
                                 float *h_pointsY,
                                 float *h_pointsZ,
                                 int   *h_occupancy)
{
	// TODO: Add some kind of size checking here
	// Assumes occupancy is a row vector
	// Assumes points is N * 3 matrix (N rows, 3 cols, 1 for each coord)
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

	// Output class copy
	std::copy(occupancyOutput, 
			  occupancyOutput + occupancy.rows(), 
			  h_occupancy);	
	
}

Eigen::MatrixXf convertWeightPointerToEigen(float *d_weights, int nWeights)
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
