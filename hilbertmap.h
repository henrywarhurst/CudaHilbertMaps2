#ifndef HILBERTMAP_H_ 
#define HILBERTMAP_H_

#include <eigen3/Eigen/Dense>

class HilbertMap
{
    public:
		HilbertMap(float lengthScale, 
				   Eigen::MatrixXf points,	
				   Eigen::MatrixXi occupancy,
				   std::vector<int> r,
				   std::vector<int> g,
				   std::vector<int> b,
				   std::vector<float> surfX,
				   std::vector<float> surfY,
				   std::vector<float> surfZ);

		void train(	float learningRate,
					float regularisationLambda);

		void trainHost( float learningRate,
						float regularisationLambda);
		
		double query(Eigen::Vector3f point);
		double queryHost(Eigen::Vector3f point);

		Eigen::MatrixXf getFeaturesHost(Eigen::Vector3f point);

		Eigen::MatrixXf getWeights() const;

		void savePoseViewToPcdCuda();
		void savePoseViewToPcd(Eigen::Matrix4f pose);

    private:
		float lengthScale_;

		Eigen::MatrixXf points_;
		Eigen::MatrixXi occupancy_;
		Eigen::MatrixXf weights_;
		
		std::vector<int> r_;
		std::vector<int> g_;
		std::vector<int> b_;

		std::vector<float> surfX_;
		std::vector<float> surfY_;
		std::vector<float> surfZ_;

		float *weightsR_;
		float *weightsG_;
		float *weightsB_;
		
};

#endif /* HILBERTMAP_H_ */ 
