#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include "common/aicallback.h"

namespace Intelligent {
using namespace std;

using ai_callback = MessageCallBackDataInfo;

class IntelligentTraffic {
public:
    // Here, uri means json_data.dump()
    // Not thread safe
    virtual bool make_view(const string &uri, size_t timeout = 100) = 0;
    virtual void set_callback(ai_callback callback)                 = 0;
    virtual void disconnect_view(const string &dis_uri)             = 0;
    virtual vector<string> get_uris() const                         = 0;
};

shared_ptr<IntelligentTraffic> create_intelligent_traffic(const string model_repository);
};  // namespace Intelligent