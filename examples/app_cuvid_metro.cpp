/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-02 15:36:07
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-15 11:16:11
 *************************************************************************************/
#include "app_metro/metro.hpp"
#include <string>
#include "common/ilogger.hpp"
#include <fstream>
#include <map>

static void callback(int callbackType, void *img, char *data, int datalen, int w, int h, int device) {
    // 这里采用calllbacktype 当做cameraid
    return;
}
static void test_metro() {
    auto instance = metro::create_solution(".", {0}, 1);
    if (instance == nullptr) {
        INFO("create instance failed.");
        return;
    }
    instance->set_callback(callback);
    std::vector<std::string> raw_datas = {
        R"({"cameraID":"1","uri":"rtsp://172.16.180.146/record/image/src/2_1.mp4","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})",
        R"({"cameraID":"2","uri":"rtsp://172.16.180.146/record/image/src/3.mp4","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})",
        R"({"cameraID":"3","uri":"rtsp://172.16.180.146/record/image/src/2_2.mp4","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})",
        R"({"cameraID":"4","uri":"rtsp://172.16.180.146/record/image/src/1.mp4","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})",
        R"({"cameraID":"5","uri":"rtsp://admin:xmrbi123@192.168.173.241:554/cam/realmonitor?ch1&1","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})",
        R"({"cameraID":"6","uri":"rtsp://admin:xmrbi123@192.168.173.242:554/cam/realmonitor?ch1&1","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})",
        R"({"cameraID":"7","uri":"rtsp://admin:12345@192.168.173.244:554/video2","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})",
        R"({"cameraID":"8","uri":"rtsp://xmrbi:xmrbi123@192.168.173.240:554/cam/realmonitor?ch1&1","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})",
        R"({"cameraID":"9","uri":"rtsp://admin:123456@192.168.173.246:554/cam/realmonitor?ch1&1","events":[{"eventName":"anjian","enable":true},{"eventName":"baojie","enable":true},{"eventName":"xingren","enable":true}]})"

    };
    int loop        = 3;
    int streams     = raw_datas.size();
    int total_views = loop * streams;
    // #pragma omp parallel for num_threads(total_views)
    for (int i = 0; i < total_views; ++i) {
        auto replaced_string = raw_datas[i % streams];
        auto success         = instance->make_view(replaced_string);
    }

    auto views = instance->get_uris();
    INFO("tatal %d streams.", views.size());
    iLogger::sleep(10000000);
}

int app_metro() {
    test_metro();

    return 0;
}