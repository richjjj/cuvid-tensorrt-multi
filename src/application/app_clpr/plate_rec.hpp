#ifndef PLATE_REC_HPP
#define PLATE_REC_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>

namespace Clpr {
using namespace std;
using namespace cv;
const vector<string> CHARS = {
    "#",  "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂",
    "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "学", "警", "港", "澳", "挂", "使",
    "领", "民", "航", "深", "0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "A",  "B",  "C",  "D",  "E",
    "F",  "G",  "H",  "J",  "K",  "L",  "M",  "N",  "P",  "Q",  "R",  "S",  "T",  "U",  "V",  "W",  "X",  "Y",  "Z"};

enum class PlateColor { black = 0, blue, green, white, yellow, unknown };
struct plateResult {
    PlateColor color{PlateColor::unknown};
    float color_confidence{0};
    string number{};
    float number_confidence{0};
    string color_str() const {
        string tmp{};
        switch (color) {
            case PlateColor::black:
                tmp = "黑";
                break;
            case PlateColor::blue:
                tmp = "蓝";
                break;
            case PlateColor::green:
                tmp = "绿";
                break;
            case PlateColor::white:
                tmp = "白";
                break;
            case PlateColor::yellow:
                tmp = "黄";
                break;
            default:
                tmp = "unknown";
                break;
        }
        return tmp;
    }
    string print_info() const {
        auto tmp = color_str();
        stringstream result{};
        result << "number: " << number << ", number_socre: " << number_confidence << ", color: " << tmp
               << ", color_score: " << color_confidence;
        return result.str();
    }
};

using RecInput = tuple<Mat, float *>;

class RecInfer {
public:
    virtual shared_future<plateResult> commit(const RecInput &input)                   = 0;
    virtual vector<shared_future<plateResult>> commits(const vector<RecInput> &inputs) = 0;
};

shared_ptr<RecInfer> create_rec(const string &engine_file, int gpuid = 0);

}  // namespace Clpr

#endif