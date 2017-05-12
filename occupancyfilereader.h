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

		std::vector<int> xPoints_;
		std::vector<int> yPoints_;
		std::vector<int> zPoints_;
		std::vector<int> oPoints_;

		Eigen::MatrixXf points_;
		Eigen::MatrixXi occupancy_;

};


#endif /* OCCUPANCYFILEREADER_H_ */
