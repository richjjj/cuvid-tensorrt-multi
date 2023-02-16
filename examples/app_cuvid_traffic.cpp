/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-02 15:36:07
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-15 11:16:11
 *************************************************************************************/
#include "app_cuvid_intelligenttrafficsurveillance/intelligent_traffic.hpp"
#include <string>
#include "common/ilogger.hpp"
static void callback(int callbackType, void *img, char *data, int datalen) {
    // INFO("data is %s.", data);
    ;
}
static void test_traffic() {
    auto instance = Intelligent::create_intelligent_traffic(".", {0, 1, 2, 3}, 1);
    if (instance == nullptr) {
        INFO("create instance failed.");
        return;
    }
    instance->set_callback(callback);
    std::string raw_data =
        R"({"cameraID":"1","uri":"rtsp://192.168.172.212/record/test1.mp4","events":[{"eventName":"nixing","enable":true,"rois":[{"roiName":"default nixing","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"xingrenchuangru","enable":true,"rois":[{"roiName":"default xingrenchuangru","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"feijidongche","enable":true,"rois":[{"roiName":"default feijidongche","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"biandao","enable":true,"rois":[{"roiName":"变道","pointsNum":2,"points":{"x1":477,"y1":368,"x2":701,"y2":870,"x3":0,"y3":0,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"weiting","enable":true,"rois":[{"roiName":"default weiting","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"eventName":"yongdu","enable":true,"rois":[{"roiName":"default yongdu","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1920,"y2":0,"x3":1920,"y3":1080,"x4":0,"y4":1080,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]}]})";

    int num_views = 2;
#pragma omp parallel for num_threads(num_views)
    for (int i = 0; i < num_views; ++i) {
        auto replaced_string = iLogger::replace_string(raw_data, "1", std::to_string(i), 1);
        // replaced_string      = iLogger::replace_string(replaced_string, "test1", "test" + std::to_string(i), 1);
        auto success = instance->make_view(replaced_string);
        while (!success) {
            // INFO("failed to make_view[test%d] and retry", i);
            success = instance->make_view(replaced_string);
            iLogger::sleep(5000);
        }
    }
    iLogger::sleep(10000000);
    // instance->stop();
}

int app_traffic() {
    test_traffic();
    return 0;
}