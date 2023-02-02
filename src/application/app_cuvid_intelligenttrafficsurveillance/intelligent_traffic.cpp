/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-01 10:12:40
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-02 10:27:51
 *************************************************************************************/
#include "intelligent_traffic.hpp"

#include "builder/trt_builder.hpp"
#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "ffhdd/cuvid-decoder.hpp"
#include "ffhdd/ffmpeg-demuxer.hpp"

namespace Intelligent {
using json = nlohmann::json;
using namespace std;

static json parse_raw_data(const string &raw_data) {
    return std::move(json::parse(raw_data));
}

class IntelligentTrafficImpl : public IntelligentTraffic {
public:
    IntelligentTrafficImpl() {}
    virtual bool make_view(const string &raw_data, size_t timeout) override {
        promise<bool> pro;
        // parse data 得到接口文档的结果
        auto j_data    = json::parse(raw_data);
        string uri     = j_data["uri"];
        runnings_[uri] = true;
        ts_.emplace_back(thread(&IntelligentTrafficImpl::worker, this, uri, ref(pro)));
        bool state = pro.get_future().get();
        if (state) {
            uris_.emplace_back(uri);
        } else {
            INFOE("The uri connection is refused.");
            runnings_[uri] = false;
        }
        return state;
    }
    virtual void worker(const string &uri, promise<bool> &state) {
        ;
    }
    virtual void set_callback(ai_callback callback) override {
        callback_ = callback;
    }
    virtual vector<string> get_uris() const override {
        return uris_;
    }
    virtual bool startup(const string &model_repository) {
        model_repository_ = model_repository;
        // load model
        // TODO
        return true;
    }

private:
    // multi gpus
    vector<int> gpus_{0};
    vector<thread> ts_;
    vector<string> uris_;
    map<string, atomic_bool> runnings_;
    ai_callback callback_;
    string model_repository_;
};
shared_ptr<IntelligentTraffic> create_intelligent_traffic(const string &model_repository) {
    shared_ptr<IntelligentTrafficImpl> instance(new IntelligentTrafficImpl());
    if (!instance->startup(model_repository)) {
        instance.reset();
    }
    return instance;
}

};  // namespace Intelligent