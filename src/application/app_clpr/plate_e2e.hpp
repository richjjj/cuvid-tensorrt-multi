#ifndef PLATE_E2E_HPP
#define PLATE_E2E_HPP
#include "opencv2/opencv.hpp"
#include <future>
#include <memory>
#include <vector>
#include <string>
namespace Clpr {
using namespace std;
using namespace cv;

using e2eInput = Mat;
struct plateInfo {
    float left, top, right, bottom, box_confidence;
    string plate_type{};  // single, multi 单或双层车牌
    // float landmarks[8];
    string number{};
    float number_confidence{0};
    string color{};
    float color_confidence{0};
};

using e2eOutput = vector<plateInfo>;
class e2eInfer {
public:
    // virtual shared_future<e2eOutput> commit(const e2eInput& input)                   = 0;
    // virtual vector<shared_future<e2eOutput>> commits(const vector<e2eInput>& inputs) = 0;
    virtual e2eOutput detect(const e2eInput& input)                   = 0;
    virtual vector<e2eOutput> detects(const vector<e2eInput>& inputs) = 0;
};
shared_ptr<e2eInfer> create_e2e(const string& det_file, const string& rec_file, float confidence_threshold = 0.5,
                                float nms_threshold = 0.4, int gpuid = 0);
}  // namespace Clpr
#endif