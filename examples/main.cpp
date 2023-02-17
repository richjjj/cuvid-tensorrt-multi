/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-01-13 13:06:23
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-02 14:02:58
 *************************************************************************************/

#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

#ifndef JETSON
int app_demuxer();
int app_hard_decode();
int app_multi_camera();
int app_pipeline();
int app_cuvid_yolo();
int app_cuvid_yolopose();
void multi_gpu_test();
int app_yolo();
int app_traffic();
int app_json();
int app_bus();
int app_test();
#endif
int app_yolopose();
int app_plate();
#ifndef JETSON
int main(int argc, char **argv) {
    const char *method = "yolo";
    if (argc > 1) {
        method = argv[1];
    }

    if (strcmp(method, "demuxer") == 0) {
        app_demuxer();
    } else if (strcmp(method, "hard_decode") == 0) {
        app_hard_decode();
    } else if (strcmp(method, "cuvid_yolo") == 0) {
        app_cuvid_yolo();
    } else if (strcmp(method, "cuvid_yolopose") == 0) {
        app_cuvid_yolopose();
    } else if (strcmp(method, "multi") == 0) {
        app_multi_camera();
    } else if (strcmp(method, "pipeline") == 0) {
        app_pipeline();
    } else if (strcmp(method, "plate") == 0) {
        app_plate();
    } else if (strcmp(method, "yolo") == 0) {
        app_yolo();
    } else if (strcmp(method, "multi_gpu") == 0) {
        multi_gpu_test();
    } else if (strcmp(method, "json") == 0) {
        app_json();
    } else if (strcmp(method, "traffic") == 0) {
        app_traffic();
    } else if (strcmp(method, "bus") == 0) {
        app_bus();
    } else if (strcmp(method, "test") == 0) {
        app_test();
    } else {
        app_yolo();
    }
    return 0;
}
#else
int main(int argc, char **argv) {
    const char *method = "yolo";
    if (argc > 1) {
        method = argv[1];
    }
    if (strcmp(method, "plate") == 0) {
        app_plate();
    } else if (strcmp(method, "yolopose") == 0) {
        app_yolopose();
    } else {
        app_plate();
    }
    return 0;
}
#endif
