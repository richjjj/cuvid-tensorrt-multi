
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
#endif
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
    } else if (strcmp(method, "plate") == 0) {
        app_plate();
    } else {
        app_plate();
    }
    return 0;
}
#endif
