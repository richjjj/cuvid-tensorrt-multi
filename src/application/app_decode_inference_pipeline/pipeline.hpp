#ifndef PIPELINE_HPP
#define PIPELINE_HPP
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include "common/aicallback.h"
namespace Pipeline {
using namespace std;
// using ai_callback = function<void(int callbackType, void *image, char *data, int datalen)>;
using ai_callback = CallBackDataInfo;

class Pipeline {
public:
    // 先 set_callback 后 make_views
    // timeout : s， if -1 ：ignore
    virtual bool make_view(const string &uri, size_t timeout = 100)                   = 0;
    virtual vector<bool> make_views(const vector<string> &uris, size_t timeout = 100) = 0;
    virtual void set_callback(ai_callback callback)                                   = 0;
    virtual void disconnect_view(const string &dis_uri)                               = 0;
    virtual void disconnect_views(const vector<string> &dis_uris)                     = 0;
    virtual vector<string> get_uris() const                                           = 0;
    virtual void join()                                                               = 0;
};
shared_ptr<Pipeline> create_pipeline(const string &det_name, const string &pose_name = "", const string &gcn_name = "",
                                     int gpuid = 0, bool use_device_frame = true);

};  // namespace Pipeline

#endif  // PIPELINE_HPP
