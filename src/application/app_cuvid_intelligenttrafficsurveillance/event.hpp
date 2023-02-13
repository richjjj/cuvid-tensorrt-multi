/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-06 15:24:28
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-10 16:21:15
 *************************************************************************************/
#pragma once
#include "common/object_detector.hpp"
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>
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
class EventInfer {
    // return：是否触发事件
    virtual bool commit(const ObjectDetector::Box& box) = 0;
};

std::shared_ptr<EventInfer> create_event(const std::string& event_name);

};  // namespace Intelligent
