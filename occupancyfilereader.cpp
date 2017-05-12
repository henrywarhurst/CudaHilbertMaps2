#include "occupancyfilereader.h"

#include <fstream>
#include <sstream>

OccupancyFileReader::OccupancyFileReader(std::string configFileName)
	: configFileName_(configFileName)
{}

void OccupancyFileReader::parse()
{
	std::string curLine;
	ifstream configFile (configFileName_);
	if (!configFile.is_open()) {
		std::cout << "Could not find config file with name: " << configFileName;
		std::cout << std::endl;	
		return;
	}	

	while (std::getline(configFile, curLine)) {
		ifstream curOccupancyFrameFile (curLine);
		
		if (!curOccupancyFrameFile.is_open()) continue;

		// Read in the current .occ file	
		std::string occLine;
		while (std::getline(curOccupancyFrameFile, occLine)) {
			std::string cell;
			std::stringstream lineStream(occLine);
			size_t columnIdx = 0;
			while (std::getline(lineStream, cell, ',')) {
				if (columnIdx == 0) {
					xPoints_.push_back(std::stoi(cell));					
				} else if (columnIdx == 1) {
					yPoints_.push_back(std::stoi(cell));	
				} else if (columnIdx == 2) {
					zPoints_.push_back(std::stoi(cell));
				} else if (columnIdx == 3) {
					oPoints_.push_back(std::stoi(cell));
				}
			}			
		}
		curOccupancyFrameFile.close();
	}
	configFile.close();

	// Convert points to Eigen matrices
	if (xPoints_.size() != yPoints_.size() || 
		yPoints_.size() != zPoints_.size() ||
		zPoints_.size() != oPoints_.size()) {

		std::cout << "Inconsitent point data! Cannot build Eigen matrices!" << std::endl;
		return;
	}
	
	size_t nPoints = xPoints_.size();
	points_.resize(nPoints, 3);
	occupancy_.resize(nPoints, 1);

	for (size_t i=0; i<nPoints, ++i) {
		points_(i, 0) = xPoints[i];
		points_(i, 1) = yPoints[i];
		points_(i, 2) = zPoints[i];
		
		occupancy_(i, 0) = oPoints[i]; 
	}
}

Eigen::MatrixXd getPoints() const
{

}

Eigen::MatrixXi getOccupancy() const
{

}
