#ifndef OCCUPANCYFILEREADER_H_ 
#define OCCUPANCYFILEREADER_H_

#include <string>
#include <fstream>
#include <vector>

#include <eigen3/Eigen/Dense>

class OccupancyFileReader
{
	public:
		OccupancyFileReader(std::string path);
		
		void parse();

		Eigen::MatrixXf getPoints() const;
		Eigen::MatrixXi getOccupancy() const;

	private:
		std::string configFileName_;

		std::vector<float> xPoints_;
		std::vector<float> yPoints_;
		std::vector<float> zPoints_;
		std::vector<int> oPoints_;
	
		std::vector<float> xSurf_;
		std::vector<float> ySurf_;
		std::vector<float> zSurf_;
	
		std::vector<int> r_;
		std::vector<int> g_;
		std::vector<int> b_;

		Eigen::MatrixXf points_;
		Eigen::MatrixXi occupancy_;
};


#endif /* OCCUPANCYFILEREADER_H_ */
