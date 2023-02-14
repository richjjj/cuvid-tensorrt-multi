/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-06 15:24:28
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-14 09:50:22
 *************************************************************************************/
#pragma once
#include "common/object_detector.hpp"
#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include <memory>
#include <string>
namespace Intelligent {
struct RoiConfig {
    std::string roiName;
    int pointsNum;
    std::vector<cv::Point2f> points;
};
struct EventConfig {
    std::string eventName;
    bool enable{false};
    std::vector<RoiConfig> rois;
};
struct ViewConfig {
    std::string cameraID;
    std::string uri;
    std::vector<EventConfig> events;
};
struct Input {
    unsigned int frame_index_{0};
    YoloGPUPtr::Image image;
    // ObjectDetector::Box box;
    YoloGPUPtr::BoxArray boxarray_;
};
class EventInfer {
    // return：是否触发事件
    virtual bool commit(const Input& input) = 0;
};

std::shared_ptr<EventInfer> create_event(const string& raw_data);

};  // namespace Intelligent
