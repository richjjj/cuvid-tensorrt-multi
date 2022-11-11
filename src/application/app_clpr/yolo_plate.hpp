#ifndef CLPR
#define CLPR

#include "opencv2/opencv.hpp"
#include <future>
#include <memory>
#include <vector>
#include <string>

namespace Clpr {
using namespace std;
using namespace cv;

struct PlateRegion {
    float left, top, right, bottom, confidence;
    int class_label;
    Point2f landmarks[4];

    PlateRegion() = default;
    PlateRegion(float left, float top, float right, float bottom, float confidence, int class_label, float* points)
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {
        memcpy(landmarks, points, 4);
    }
    // float d2i[6];
};
using PlateRegionArray = vector<PlateRegion>;

class DetInfer {
public:
    virtual shared_future<PlateRegionArray> commit(const Mat& image)                   = 0;
    virtual vector<shared_future<PlateRegionArray>> commits(const vector<Mat>& images) = 0;
};
shared_ptr<DetInfer> create_det(const string& engine_file, int gpuid = 0, float confidence_threshold = 0.5,
                                float nms_threshold = 0.4, int max_objects = 1024);
}  // namespace Clpr

#endif