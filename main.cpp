#include <iostream>
#include <eigen3/Eigen/Dense>
#include "./Cuda/logisticregression.cuh"

int main()
{
	float lengthScale = 1.0f;
	float learningRate = 0.001f;
	float regularisationLambda = 1.0f;

	Eigen::MatrixXf points = Eigen::MatrixXf::Random(50, 3);
	Eigen::MatrixXi occupancy = Eigen::MatrixXi::Zero(50, 1);	
	//Eigen::MatrixXf weights = runLogisticRegression(points, 
	//												occupancy,
	//												lengthScale,
	//												learningRate,
	//												regularisationLambda);
	
	print_hi();
	std::cout << MAX_THREADS << std::endl;
	return 0;
}
