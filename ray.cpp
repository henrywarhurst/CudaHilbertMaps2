#include "ray.h"

#include <iostream>

Ray::Ray(size_t imageU, size_t imageV)
{
	for (size_t rayZ=kStartZ; rayZ<kMaxZ; rayZ+=kStepZ) {
		float curZ = (float) rayZ / kCorrectionFactor;
		float curX = (imageU - kOpticalCentreX)*curZ / kFocalLengthX;
		float curY = (imageV - kOpticalCentreY)*curZ / kFocalLengthY;

		Eigen::Vector3f curPoint;
		curPoint << curX, curY, curZ;
		pointsOriginal_.push_back(curPoint);
	}
	pointsTransformed_ = pointsOriginal_;
}	

size_t Ray::size() 
{
	return pointsOriginal_.size();
}

std::vector<Eigen::Vector3f> Ray::getPoints() const
{
	return pointsTransformed_;
}

void Ray::transformToPose(Eigen::Matrix4f pose)
{
	pointsTransformed_ = pointsOriginal_;
	for (auto &point : pointsTransformed_) {
		Eigen::Vector4f tmpPoint;
		tmpPoint << point(0), point(1), point(2), 1;
		Eigen::Vector4f transformedPoint = pose * tmpPoint;
		point(0) = transformedPoint(0);
		point(1) = transformedPoint(1);
		point(2) = transformedPoint(2);
	}
}

void Ray::setSurfaceIntersectionPoint(Eigen::Vector3f surfaceIntersectionPoint)
{
	surfaceIntersection_ = surfaceIntersectionPoint;
}

Eigen::Vector3f Ray::getSurfaceIntersectionPoint() const
{
	return surfaceIntersection_;
}
