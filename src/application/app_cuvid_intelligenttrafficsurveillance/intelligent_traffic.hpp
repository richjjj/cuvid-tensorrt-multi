/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-01 09:54:07
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-02 16:34:45
 *************************************************************************************/
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
    virtual bool make_view(const string& uri, size_t timeout = 100) = 0;
    virtual void set_callback(ai_callback callback)                 = 0;
    virtual void stop()                                             = 0;
    // virtual void disconnect_view(const string &dis_uri)             = 0;
    virtual vector<string> get_uris() const = 0;
};

shared_ptr<IntelligentTraffic> create_intelligent_traffic(const string& model_repository,
                                                          const vector<int> gpuids = {0, 1, 2, 3});
};  // namespace Intelligent