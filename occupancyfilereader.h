#ifndef OCCUPANCYFILEREADER_H_ 
#define OCCUPANCYFILEREADER_H_

#include <string>
#include <fstream>

#include <eigen3/Eigen/Dense>

class OccupancyFileReader
{
	public:
		OccupancyFileReader(std::string path);
		
		void parse();

		Eigen::MatrixXd getPoints() const;
		Eigen::MatrixXi getOccupancy() const;

	private:
		std::string path_;

		Eigen::MatrixXd points_;
		Eigen::MatrixXi occupancy_;

};


#endif /* OCCUPANCYFILEREADER_H_ */

