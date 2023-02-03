/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-02 15:36:07
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-02 17:00:33
 *************************************************************************************/
#include "app_cuvid_intelligenttrafficsurveillance/intelligent_traffic.hpp"
#include <string>
#include "common/ilogger.hpp"
static void test_traffic() {
    auto instance = Intelligent::create_intelligent_traffic(".", {0, 1, 2, 3});
    std::string raw_data =
        R"({"cameraID":"1","uri":"exp/35.mp4","events":[{"EventName":"拥堵","enable":true,"roi":[{"roiName":"area_1","pointsNum":4,"points":{"x1":1,"y1":2,"x2":0,"y2":0,"x3":0,"y3":0,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"EventName":"停车","enable":true,"roi":[{"roiName":"area_2","pointsNum":4,"points":{"x1":3,"y1":0,"x2":0,"y2":0,"x3":0,"y3":1,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"EventName":"应急车道","enable":true,"roi":[{"roiName":"area_3","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1,"y2":0,"x3":0,"y3":2,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}},{"roiName":"area_5","pointsNum":4,"points":{"x1":0,"y1":0,"x2":3,"y2":0,"x3":2,"y3":0,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"EventName":"变道，压线","enable":true,"roi":[{"roiName":"area_4","pointsNum":2,"points":{"x1":0,"y1":0,"x2":2,"y2":0,"x3":0,"y3":0,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]}]})";
    for (int i = 0; i < 150; ++i) {
        instance->make_view(raw_data);
    }
    iLogger::sleep(200);
    instance->stop();
}

int app_traffic() {
    test_traffic();
    return 0;
}