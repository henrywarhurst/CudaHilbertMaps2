#ifndef HILBERTMAP_H_ 
#define HILBERTMAP_H_

#include <eigen3/Eigen/Dense>

class HilbertMap
{
    public:
		HilbertMap(float lengthScale);

		void train(	Eigen::MatrixXf points, 
					Eigen::MatrixXi occupancy, 
					float learningRate,
					float regularisationLambda);
		
		double query(Eigen::Vector3f point);

		void savePoseViewToPcd(Eigen::Matrix4f pose);

    private:
		float lengthScale_;
		Eigen::MatrixXf weights_;
		
};

#endif /* HILBERTMAP_H_ */ 
