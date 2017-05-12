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
#include "hilbertmap.h"

#include "Cuda/logisticregression.cuh"

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

void testHilbertMap1()
{
	float lengthScale = 1.0f;
	float learningRate = 0.001f;
	float regularisationLambda = 1.0f;
	size_t numPoints = 1000000;

	Eigen::MatrixXf points = Eigen::MatrixXf::Random(numPoints, 3);
	Eigen::MatrixXi occupancy = Eigen::MatrixXi::Zero(numPoints, 1);	
	
	HilbertMap hm(lengthScale, points, occupancy);

	hm.train(learningRate, regularisationLambda);

	Eigen::MatrixXf weights = hm.getWeights();
}

void testHilbertMap2()
{
	float lengthScale = 1.0f;
	float learningRate = 0.001f;
	float regularisationLambda = 1.0f;

	OccupancyFileReader fileReader("/home/henry/ICPCUDA_results/run2/config.ini");
	
	fileReader.parse();
	Eigen::MatrixXf points = fileReader.getPoints();
	Eigen::MatrixXi occupancy = fileReader.getOccupancy();
	
	HilbertMap hm(lengthScale, points, occupancy);
	hm.train(learningRate, regularisationLambda);
}

void testHilbertMap3()
{
	float lengthScale = 1.0f;
	float learningRate = 0.001f;
	float regularisationLambda = 1.0f;

	OccupancyFileReader fileReader("/home/henry/ICPCUDA_results/run2/config.ini");
	
	fileReader.parse();
	Eigen::MatrixXf points = fileReader.getPoints();
	Eigen::MatrixXi occupancy = fileReader.getOccupancy();
	
	HilbertMap hm(lengthScale, points, occupancy);
	hm.train(learningRate, regularisationLambda);
	
	Eigen::Vector3f queryPoint;
	queryPoint << -0.83369,-0.594651,1.50173;

	float occupancyPrediction = hm.query(queryPoint);
	std::cout << "Probability of occupancy is " << occupancyPrediction << std::endl;
}

int main()
{
	//testHilbertMap1();
	//testHilbertMap2();
	testHilbertMap3();
	
	return 0;
}
