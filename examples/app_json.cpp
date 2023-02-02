/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-01-17 16:34:04
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-02 15:46:13
 *************************************************************************************/
#include "common/json.hpp"
#include <string>
#include <iostream>
int app_json() {
    using json = nlohmann::json;
    json ex1   = json::parse(
        R"({"cameraID":"1","uri":"2222","events":[{"EventName":"拥堵","enable":true,"roi":[{"roiName":"area_1","pointsNum":4,"points":{"x1":1,"y1":2,"x2":0,"y2":0,"x3":0,"y3":0,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"EventName":"停车","enable":true,"roi":[{"roiName":"area_2","pointsNum":4,"points":{"x1":3,"y1":0,"x2":0,"y2":0,"x3":0,"y3":1,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"EventName":"应急车道","enable":true,"roi":[{"roiName":"area_3","pointsNum":4,"points":{"x1":0,"y1":0,"x2":1,"y2":0,"x3":0,"y3":2,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}},{"roiName":"area_5","pointsNum":4,"points":{"x1":0,"y1":0,"x2":3,"y2":0,"x3":2,"y3":0,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]},{"EventName":"变道，压线","enable":true,"roi":[{"roiName":"area_4","pointsNum":2,"points":{"x1":0,"y1":0,"x2":2,"y2":0,"x3":0,"y3":0,"x4":0,"y4":0,"x5":0,"y5":0,"x6":0,"y6":0,"x7":0,"y7":0,"x8":0,"y8":0}}]}]})");
    std::cout << ex1 << "\n";
    return 0;
}