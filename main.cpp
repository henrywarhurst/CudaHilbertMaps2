#include <iostream>
#include <eigen3/Eigen/Dense>
#include "./Cuda/logisticregression.cuh"

int main()
{
	float lengthScale = 1.0f;
	float learningRate = 0.001f;
	float regularisationLambda = 1.0f;
	size_t numPoints = 83000;

	Eigen::MatrixXf points = Eigen::MatrixXf::Random(numPoints, 3);
	Eigen::MatrixXi occupancy = Eigen::MatrixXi::Zero(numPoints, 1);	
	Eigen::MatrixXf weights = runLogisticRegression(points, 
													occupancy,
													lengthScale,
													learningRate,
													regularisationLambda);
	
	return 0;
}
