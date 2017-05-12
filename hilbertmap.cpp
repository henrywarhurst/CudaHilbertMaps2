#include "./Cuda/logisticregression.cuh"
#include "hilbertmap.h"

HilbertMap::HilbertMap(float lengthScale)
	: lengthScale_(lengthScale)
{}


void HilbertMap::train( Eigen::MatrixXf points,
						Eigen::MatrixXi occupancy,
						float learningRate,
						float regularisationLambda)
{
	// Cuda call to train the classifier
    weights_ = runLogisticRegression( points,
						              occupancy,
						              lengthScale_,
						              learningRate,
						              regularisationLambda);
}

double HilbertMap::query(Eigen::Vector3f point)
{
	return 0.0;	

}

void savePoseViewToPcd(Eigen::Matrix4f pose)
{

}
