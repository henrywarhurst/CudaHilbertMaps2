#ifndef RAY_H_ 
#define RAY_H_

#include <vector>

#include <eigen3/Eigen/Dense>

class Ray
{
	public:
		Ray(size_t imageU, size_t imageV);

		size_t size();

		void transformToPose(Eigen::Matrix4f pose);

		std::vector<Eigen::Vector3f> getPoints() const;

		void setSurfaceIntersectionPoint(Eigen::Vector3f surfaceIntersectionPoint);
		Eigen::Vector3f getSurfaceIntersectionPoint() const;

	private:
		std::vector<Eigen::Vector3f> pointsOriginal_;
		std::vector<Eigen::Vector3f> pointsTransformed_;
		Eigen::Vector3f surfaceIntersection_;

        static constexpr const double kCorrectionFactor = 1000.0    ;
        static constexpr const double kFocalLengthX     = 525.0     ;
        static constexpr const double kFocalLengthY     = 525.0     ;
        static constexpr const double kOpticalCentreX   = 319.5     ;
        static constexpr const double kOpticalCentreY   = 239.5     ;

        static constexpr const size_t kMaxZ             = 1400      ;
        static constexpr const size_t kStepZ            = 20      	;
        static constexpr const size_t kStartZ           = 700      	;
};

#endif /* RAY_H_ */
