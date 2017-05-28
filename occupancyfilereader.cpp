#include "occupancyfilereader.h"

#include <fstream>
#include <sstream>
#include <iostream>

OccupancyFileReader::OccupancyFileReader(std::string configFileName)
	: configFileName_(configFileName)
{}

void OccupancyFileReader::parse()
{
	std::string curLine;
	std::ifstream configFile (configFileName_);
	if (!configFile.is_open()) {
		std::cout << "Could not find config file with name: " << configFileName_;
		std::cout << std::endl;	
		return;
	}	

	while (std::getline(configFile, curLine)) {
		std::ifstream curOccupancyFrameFile (curLine);
		
		if (!curOccupancyFrameFile.is_open()) continue;

		std::cout << "Current liine = " << curLine << std::endl;
		// Read in the current .occ file	
		std::string occLine;
		while (std::getline(curOccupancyFrameFile, occLine)) {
			std::string cell;
			std::stringstream lineStream(occLine);
			size_t columnIdx = 0;
			float curX = 0;
			float curY = 0;
			float curZ = 0;
			while (std::getline(lineStream, cell, ',')) {
				if (columnIdx == 0) {
					curX = std::stof(cell);
					xPoints_.push_back(curX);					
				} else if (columnIdx == 1) {
					curY = std::stof(cell);
					yPoints_.push_back(curY);	
				} else if (columnIdx == 2) {
					curZ = std::stof(cell);
					zPoints_.push_back(curZ);
				} else if (columnIdx == 3) {
					oPoints_.push_back(std::stoi(cell));

				// Add colour information and store surface points
				} else if (columnIdx == 4) {
					r_.push_back(std::stoi(cell));					
				} else if (columnIdx == 5) {
					g_.push_back(std::stoi(cell));	
				} else if (columnIdx == 6) {
					b_.push_back(std::stoi(cell));
					// Associate the surface points with the colours
					xSurf_.push_back(curX);
					ySurf_.push_back(curY);
					zSurf_.push_back(curZ);
				} 
				columnIdx++;
			}			
		}
		curOccupancyFrameFile.close();
	}
	configFile.close();

	// Convert points to Eigen matrices
	if (xPoints_.size() != yPoints_.size() || 
		yPoints_.size() != zPoints_.size() ||
		zPoints_.size() != oPoints_.size()) {

		std::cout << "Inconsistent point data! Cannot build Eigen matrices!" << std::endl;
		return;
	}
	
	size_t nPoints = xPoints_.size();
	points_.resize(nPoints, 3);
	occupancy_.resize(nPoints, 1);

	for (size_t i=0; i<nPoints; ++i) {
		points_(i, 0) = xPoints_[i];
		points_(i, 1) = yPoints_[i];
		points_(i, 2) = zPoints_[i];
		
		occupancy_(i, 0) = oPoints_[i]; 
	}
}

Eigen::MatrixXf OccupancyFileReader::getPoints() const
{
	return points_;
}

Eigen::MatrixXi OccupancyFileReader::getOccupancy() const
{
	return occupancy_;
}

std::vector<float> OccupancyFileReader::getXSurf() const
{
	return xSurf_;
}

std::vector<float> OccupancyFileReader::getYSurf() const
{
	return ySurf_;
}

std::vector<float> OccupancyFileReader::getZSurf() const
{
	return zSurf_;
}

std::vector<int> OccupancyFileReader::getR() const
{
	return r_;
}

std::vector<int> OccupancyFileReader::getG() const
{
	return g_;
}

std::vector<int> OccupancyFileReader::getB() const
{
	return b_;
}

