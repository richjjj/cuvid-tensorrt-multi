/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-06 15:24:48
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-06 15:49:28
 *************************************************************************************/

#include "event.hpp"

namespace Intelligent {
using namespace std;
class EventImpl {
public:
    virtual bool startup(const std::string& event_name) {
        eventName_ = event_name;
        return true;
    }

protected:
private:
    string eventName_;
};
class BalancedImpl : public EventImpl, public Event {
public:
    virtual bool commit(const ObjectDetector::Box& box) override {
        ;
    }

private:
};
std::shared_ptr<Event> create_event(const std::string& event_name) {
    shared_ptr<Event> instance(new BalancedImpl());
    auto impl = dynamic_pointer_cast<EventImpl>(instance);
    if (!impl->startup(event_name)) {
        instance.reset();
    }
    return instance;
}

};  // namespace Intelligent