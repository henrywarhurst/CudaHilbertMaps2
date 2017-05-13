#include <cmath>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "Cuda/logisticregression.cuh"
#include "hilbertmap.h"
#include "ray.h"

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

	//Multiply the features with the weights to get the occupancy prediction
	Eigen::MatrixXf queryResult = features * weights_.transpose();

	//std::cout << "queryResults is " << queryResult(0,0) << std::endl;
	float probability = 1/(1 + exp(-queryResult(0,0)));
	
	return probability;
}

double HilbertMap::queryHost(Eigen::Vector3f point)
{
	Eigen::MatrixXf features = getFeaturesHost(point);

	Eigen::MatrixXf queryResult = features * weights_.transpose();
	
	float probability = 1/(1 + exp(-queryResult(0,0)));
	return probability;
}

Eigen::MatrixXf HilbertMap::getFeaturesHost(Eigen::Vector3f point)
{
	size_t nFeatures = weights_.size();
	Eigen::MatrixXf features(1, nFeatures);	
	for (size_t i=0; i<nFeatures; ++i) {
		float diff = (points_(i,0) - point(0)) * (points_(i,0) - point(0)) +
					 (points_(i,1) - point(1)) * (points_(i,1) - point(1)) +
					 (points_(i,2) - point(2)) * (points_(i,2) - point(2));

		features(0,i) = exp(-lengthScale_*diff);
	}
	return features;
}

void HilbertMap::savePoseViewToPcd(Eigen::Matrix4f pose)
{
	size_t width 	= 640;
	size_t height 	= 480;
	
	std::vector<Eigen::Vector3f> cloudPoints;

	for (size_t v=0; v<height; ++v) {
		for (size_t u=0; u<width; ++u) {
			size_t depthIdx = v*height + u;
			if (depthIdx % 50) continue;
			Eigen::Vector3f cloudPoint;
			cloudPoint << 0, 0, 0;
			Ray curRay(u, v);
			curRay.transformToPose(pose);
			std::vector<Eigen::Vector3f> curPoints = curRay.getPoints();
			for (auto &curPoint : curPoints) {
				double curOccupancyProbability =  queryHost(curPoint);
				if (curOccupancyProbability > 0.5) {
					cloudPoint << curPoint(0), curPoint(1), curPoint(2);					
					break;
				}
			}
			std::cout << "\r" << "u = " << u << " v = " << v;
			std::cout << " X = " << cloudPoint(0) << " Y = " << cloudPoint(1) << " Z = " << cloudPoint(2);
            std::cout << std::flush;

			cloudPoints.push_back(cloudPoint);
		}
	}

	pcl::PointCloud<pcl::PointXYZ> cloud;

	cloud.width = cloudPoints.size();
	cloud.height = 1;
	cloud.is_dense = false;
	cloud.points.resize(cloud.width * cloud.height);

	for (size_t i=0; i<cloudPoints.size(); ++i) {
		Eigen::Vector3f curCloudPoint = cloudPoints[i];
		cloud.points[i].x = curCloudPoint(0);
		cloud.points[i].y = curCloudPoint(1);
		cloud.points[i].z = curCloudPoint(2);
	}
	pcl::io::savePCDFileASCII("hilbertview.pcd", cloud);
}
