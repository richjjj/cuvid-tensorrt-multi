/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-06 15:24:28
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-06 15:33:22
 *************************************************************************************/
#pragma once
#include "common/object_detector.hpp"
#include <memory>
#include <string>
namespace Intelligent {
class Event {
    // return：是否触发事件
    virtual bool commit(const ObjectDetector::Box& box) = 0;
};

std::shared_ptr<Event> create_event(const std::string& event_name);

};  // namespace Intelligent
