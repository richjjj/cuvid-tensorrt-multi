#ifndef CLPR
#define CLPR

#include "opencv2/opencv.hpp"
#include <future>
#include <memory>
#include <vector>
#include <string>
#include "common/trt_tensor.hpp"

namespace Clpr {
using namespace std;
using namespace cv;

struct PlateRegion {
    float left, top, right, bottom, confidence;
    int class_label;
    // Point2f landmarks[4];
    float landmarks[8];

    PlateRegion() = default;
    PlateRegion(float left, float top, float right, float bottom, float confidence, int class_label, float* points)
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {
        memcpy(landmarks, points, sizeof(landmarks));
    }
    // float d2i[6];
};
using PlateRegionArray = vector<PlateRegion>;
class DetInput {
public:
    Mat image;     // 原始图片
    Rect roiRect;  // 感兴趣区域
    DetInput() = default;
    DetInput(const Mat& image) : image(image), roiRect(Rect(0, 0, image.cols, image.rows)){};
    DetInput(const Mat& image, const vector<Point>& roi_polygon) : image(image) {
        [this](const vector<Point>& p) {
            int y_collect[4] = {p[0].y, p[1].y, p[2].y, p[3].y};
            int x_collect[4] = {p[0].x, p[1].x, p[2].x, p[3].x};
            int left         = int(*std::min_element(x_collect, x_collect + 4));
            int right        = int(*std::max_element(x_collect, x_collect + 4));
            int top          = int(*std::min_element(y_collect, y_collect + 4));
            int bottom       = int(*std::max_element(y_collect, y_collect + 4));
            this->roiRect    = Rect(left, top, right - left, bottom - top);
        }(roi_polygon);
    };
    DetInput(const Mat& image, const Rect& roi_rect) : image(image), roiRect(roi_rect){};
};
class DetInfer {
public:
    virtual shared_future<PlateRegionArray> commit(const DetInput& image)                   = 0;
    virtual vector<shared_future<PlateRegionArray>> commits(const vector<DetInput>& images) = 0;
    virtual vector<shared_future<PlateRegionArray>> commits(const vector<Mat>& images)      = 0;
};
shared_ptr<DetInfer> create_det(const string& engine_file, int gpuid = 0, float confidence_threshold = 0.5,
                                float nms_threshold = 0.4, int max_objects = 1024);
void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);
}  // namespace Clpr

#endif