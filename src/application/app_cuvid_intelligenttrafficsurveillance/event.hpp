/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-06 15:24:28
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-14 10:11:23
 *************************************************************************************/
#pragma once
#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include <memory>
#include <string>
#include "common/aicallback.h"

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
    Input() = default;
    unsigned int frame_index_{0};
    YoloGPUPtr::Image image;
    YoloGPUPtr::BoxArray boxarray_;
};
using ai_callback = MessageCallBackDataInfo;

class EventInfer {
public:
    virtual bool commit(const Input& input)         = 0;
    virtual void set_callback(ai_callback callback) = 0;
    virtual std::string get_uri() const             = 0;
};

std::shared_ptr<EventInfer> create_event(const std::string& raw_data);

};  // namespace Intelligent
