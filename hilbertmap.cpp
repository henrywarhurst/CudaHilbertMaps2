#include "Cuda/logisticregression.cuh"
#include "hilbertmap.h"

HilbertMap::HilbertMap(float lengthScale, Eigen::MatrixXf points, Eigen::MatrixXi occupancy)
	: lengthScale_(lengthScale),
	  points_(points),
	  occupancy_(occupancy)
{}


void HilbertMap::train(	float learningRate,
						float regularisationLambda)
{
	// Cuda call to train the classifier
    weights_ = runLogisticRegression( points_,
						              occupancy_,
						              lengthScale_,
						              learningRate,
						              regularisationLambda);
	
}

Eigen::MatrixXf HilbertMap::getWeights() const
{
	return weights_;
}

double HilbertMap::query(Eigen::Vector3f point)
{
	Eigen::MatrixXf features = getFeatures(point, points_, lengthScale_);
	
	//TODO: Delete this line
	return 69;
}

void savePoseViewToPcd(Eigen::Matrix4f pose)
{

}
