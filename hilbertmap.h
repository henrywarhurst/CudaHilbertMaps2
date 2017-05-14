#ifndef HILBERTMAP_H_ 
#define HILBERTMAP_H_

#include <eigen3/Eigen/Dense>

class HilbertMap
{
    public:
		HilbertMap(float lengthScale, Eigen::MatrixXf points, Eigen::MatrixXi occupancy);

		void train(	float learningRate,
					float regularisationLambda);

		void trainHost( float learningRate,
						float regularisationLambda);
		
		double query(Eigen::Vector3f point);
		double queryHost(Eigen::Vector3f point);

		Eigen::MatrixXf getFeaturesHost(Eigen::Vector3f point);

		Eigen::MatrixXf getWeights() const;

		void savePoseViewToPcd(Eigen::Matrix4f pose);

    private:
		float lengthScale_;
		Eigen::MatrixXf points_;
		Eigen::MatrixXi occupancy_;
		Eigen::MatrixXf weights_;
		
};

#endif /* HILBERTMAP_H_ */ 
