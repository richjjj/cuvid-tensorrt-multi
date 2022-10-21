#ifndef PIPELINE_HPP
#define PIPELINE_HPP
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "common/aicallback.h"
namespace Pipeline
{
    using namespace std;
    // using ai_callback = function<void(int callbackType, void *image, char *data, int datalen)>;
    using ai_callback = CallBackDataInfo;

    class Pipeline
    {
    public:
        // 先 set_callback 后 make_views
        virtual void make_views(const vector<string> &uris) = 0;
        virtual void set_callback(ai_callback callback) = 0;
        virtual void disconnect_views(const vector<string> &dis_uris) = 0;
        virtual void get_uris(vector<string> &current_uris) const = 0;
        virtual void join() = 0;
    };
    shared_ptr<Pipeline> create_pipeline(const string &engine_file, int gpuid = 0, bool use_device_frame = true);

};

#endif // PIPELINE_HPP
