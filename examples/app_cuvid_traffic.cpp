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
#include <fstream>
#include <map>
// std::vector<std::ofstream> outFiles(1);
// std::map<int, std::ofstream> outFiles;
static void callback(int callbackType, void *img, char *data, int datalen) {
    // 这里采用calllbacktype 当做cameraid
    return;
    std::ofstream outFile("output.txt", std::ios::app);
    if (outFile.is_open()) {
        // 将字符串写入文件
        outFile << data << "\n";
        outFile.close();
    };
}
static void test_traffic() {
    auto instance = Intelligent::create_intelligent_traffic(".", {0}, 1);
    if (instance == nullptr) {
        INFO("create instance failed.");
        return;
    }
    instance->set_callback(callback);
    std::string raw_data =
        R"({"cameraID":"3","uri":"/cangku_workspace/videos/1_3.16.weiting.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"xingrenchuangru","enable":true}]})";
    std::vector<std::string> raw_datas = {
        // R"({"cameraID":"1","uri":"/cangku_workspace/videos/1_3.16.weiting.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"feijidongche","enable":true},{"eventName":"xingrenchuangru","enable":true}]})",
        // R"({"cameraID":"2","uri":"/cangku_workspace/videos/2_3.6.weiting.xingren.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"feijidongche","enable":true},{"eventName":"xingrenchuangru","enable":true}]})",
        // R"({"cameraID":"3","uri":"/cangku_workspace/videos/3_3.20.motuoche.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"feijidongche","enable":true},{"eventName":"xingrenchuangru","enable":true}]})",
        // R"({"cameraID":"4","uri":"/cangku_workspace/videos/4_3.21.xingren.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"feijidongche","enable":true},{"eventName":"xingrenchuangru","enable":true}]})",
        // R"({"cameraID":"5","uri":"/cangku_workspace/videos/5_3.22.xingren.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"feijidongche","enable":true},{"eventName":"xingrenchuangru","enable":true}]})",
        R"({"cameraID":"6","uri":"/cangku_workspace/videos/6_A34.motuoche.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"feijidongche","enable":true},{"eventName":"xingrenchuangru","enable":true}]})",
        // R"({"cameraID":"7","uri":"/cangku_workspace/videos/7_3.17.weiting.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"feijidongche","enable":true},{"eventName":"xingrenchuangru","enable":true}]})",
        // R"({"cameraID":"8","uri":"/cangku_workspace/videos/8_3.17.weiting.xingren.mp4","events":[{"eventName":"weiting","enable":true},{"eventName":"feijidongche","enable":true},{"eventName":"xingrenchuangru","enable":true}]})"

    };
    int num_views = raw_datas.size();
#pragma omp parallel for num_threads(num_views)
    for (int i = 0; i < num_views; ++i) {
        // auto replaced_string = iLogger::replace_string(raw_data, "1", std::to_string(i), 1);
        auto replaced_string = raw_datas[i];
        auto success         = instance->make_view(replaced_string);
        while (!success) {
            INFO("failed to make_view[test%d] and retry", i);
            success = instance->make_view(replaced_string);
            iLogger::sleep(5000);
        }
    }
    iLogger::sleep(10000000);
    // instance->join();
}

int app_traffic() {
    test_traffic();
    // if (outFile.is_open()) {
    //     // 关闭文件
    //     outFile.close();
    // }
    return 0;
}