
#include <stdio.h>
#include <string.h>
#include <common/ilogger.hpp>
#include <functional>

int app_demuxer();
int app_hard_decode();
int app_yolo();
int app_yolopose();
int app_multi_camera();
int app_pipeline();

int main(int argc, char **argv) {
    const char *method = "yolo";
    if (argc > 1) {
        method = argv[1];
    }

    if (strcmp(method, "demuxer") == 0) {
        app_demuxer();
    } else if (strcmp(method, "hard_decode") == 0) {
        app_hard_decode();
    } else if (strcmp(method, "yolo") == 0) {
        app_yolo();
    } else if (strcmp(method, "yolopose") == 0) {
        app_yolopose();
    } else if (strcmp(method, "multi") == 0) {
        app_multi_camera();
    } else if (strcmp(method, "pipeline") == 0) {
        app_pipeline();
    }
    return 0;
}
