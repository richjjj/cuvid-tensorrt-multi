#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include "common/aicallback.h"

namespace YoloGPUPtr {
class Image;
};
typedef YoloGPUPtr::Image* GPUImage_ptr;

// 分配device内存的接口
void f1(void** gpu_ptr, size_t count);
// 拷贝device数据的接口
void f2(void* gpu_src, void* gpu_dst, size_t count);
// device to host
void f3(void* gpu_src, void* cpu_dst, size_t count);

// gpu2cpu
// gpu_ptr：允许显式地通过reinterpret_cast将void* 转成GPUImage_ptr
// cpu_ptr: 已经分配了的cpu内存地址
void convert_GPUImage_to_CPUImage(void* gpu_ptr, unsigned char* cpu_ptr, int& image_height, int& image_width);

namespace metro {
using namespace std;

using ai_callback = MessageCallBackDataInfo;

class Solution {
public:
    // Here, raw_data means json_data.dump()
    // Not thread safe
    virtual bool make_view(const string& raw_data, size_t timeout = 100) = 0;
    virtual void set_callback(ai_callback callback)                      = 0;
    // stop 所有视频流
    virtual void stop() = 0;
    // 等待所有流完成
    // virtual void join() = 0;
    // 停止指定视频流，dis_uri 为rtsp流地址
    virtual void disconnect_view(const string& dis_uri) = 0;
    // 获取当前所有的视频流
    virtual vector<string> get_uris() const = 0;
};

shared_ptr<Solution> create_solution(const string& model_repository, const vector<int> gpuids = {0, 1, 2, 3},
                                     int instances_per_device = 2);
};  // namespace metro