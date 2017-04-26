#include <eigen3/Eigen/Core>
#include "logisticregression.cuh"


// Should take in datapoints as eigen matrix (N x 3) input
Eigen::MatrixXf runLogisticRegression(const Eigen::MatrixXd &points,
                                      const Eigen::VectorXi &occupancy,
									  float learningRate,
									  float regularisationLambda)
{
	Eigen::MatrixXf tmp(1,1);
	return tmp;	
}

int getNumBlocks(int numDataPoints)
{
	return (int) ceil((float) numDataPoints / (double) MAX_THREADS);
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

Eigen::MatrixXf convertWeightPointerToEigen(float *h_weights, int nWeights)
{
	return Eigen::Map<Eigen::MatrixXf>(h_weights, 1, nWeights);
}


__global__ void cudaRbf(float *d_x, float *d_y, float *d_z,
                        float *outputFeatures, int *d_pointIdx,
                        float *d_lengthScale)
{
    int cudaIdx = threadIdx.x + blockIdx.x * blockDim.x;

    float diff = (d_x[cudaIdx] - d_x[*d_pointIdx]) * (d_x[cudaIdx] - d_x[*d_pointIdx]) +
                 (d_y[cudaIdx] - d_y[*d_pointIdx]) * (d_y[cudaIdx] - d_y[*d_pointIdx]) +
                 (d_z[cudaIdx] - d_z[*d_pointIdx]) * (d_z[cudaIdx] - d_z[*d_pointIdx]);

	outputFeatures[cudaIdx] = (float) exp(-*d_lengthScale * diff);
}

__global__ void cudaSgd(int *d_occupancy,
                        float *d_weights,
                        float *d_features,
                        int *d_pointIdx,
                        curandState_t *states,
						float learningRate,
						float lambda)
{
    int cudaIdx = threadIdx.x + blockIdx.x * blockDim.x;

    // If this is is the first example, just initialise the weights
	if (*d_pointIdx == 0) {
	      // Random value between 0 and 1
	      d_weights[cudaIdx] = (float) (curand(&states[blockIdx.x]) % 1000) / 1000.0; 
	} else {
		float numerator 	= -d_occupancy[*d_pointIdx] * d_features[cudaIdx];
    	float denominator 	= 1 + exp(-numerator*d_weights[cudaIdx]);         

        // Just using L2 regularisation here, may use elastic net later
        float regulariser = lambda*d_weights[cudaIdx];

        // Combine all the parts
        float lossGradient = (numerator/denominator) + regulariser;

        // Update weight
        d_weights[cudaIdx] = d_weights[cudaIdx] - learningRate*lossGradient;
	}
}

