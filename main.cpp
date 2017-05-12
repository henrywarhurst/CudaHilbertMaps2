/*
 * This file is part of CudaHilbertMaps.
 *
 * Copyright (C) 2017 University of Sydney.
 *
 * The use of the code within this file and all code
 * within files that make up the software that is
 * CudaHilbertMaps is permitted for non-commercial
 * purposes only. The full terms and conditions that
 * apply to the code within this file are detailed
 * within the LICENSE.txt file unless explicitly
 * stated. By downloading this file you agree to
 * comply with these terms.
 *
 * Author: Henry Warhurst
 *
 */

#include <iostream>
#include <eigen3/Eigen/Dense>

#include "occupancyfilereader.h"

#include "./Cuda/logisticregression.cuh"

int testMillionPoints()
{
	float lengthScale = 1.0f;
	float learningRate = 0.001f;
	float regularisationLambda = 1.0f;
	size_t numPoints = 1000000;

	Eigen::MatrixXf points = Eigen::MatrixXf::Random(numPoints, 3);
	Eigen::MatrixXi occupancy = Eigen::MatrixXi::Zero(numPoints, 1);	
	Eigen::MatrixXf weights = runLogisticRegression(points, 
													occupancy,
													lengthScale,
													learningRate,
													regularisationLambda);
	
	return 0;
}

int main()
{
	OccupancyFileReader fileReader("/home/henry/ICPCUDA_results/run1/config.ini");	

	fileReader.parse();
	Eigen::MatrixXd points = fileReader.getPoints();
	Eigen::MatrixXi occupancy = fileReader.getOccupancy();

	std::cout << points << std::endl;

	std::cout << occupancy << std::endl;
	
	return 0;
}
