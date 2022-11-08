#include "app_decode_inference_pipeline/pipeline.hpp"
// #include "common/ilogger.hpp"
#include <iostream>
// #include <opencv2/opencv.hpp>

void callback(int callbackType, void *img, char *data, int datalen) {
    // std::cout << "callbackType is " << callbackType << std::endl;
    // std::cout << "datalen is : " << datalen << std::endl;
    std::cout << "data is: " << data << std::endl;
}
void test_pipeline() {
    // iLogger::rmtree("imgs");
    // iLogger::mkdir("imgs");
    std::string det_name  = "yolov5x-aqm";
    std::string pose_name = "yolov5s_pose";
    std::string gcn_name  = "";
    std::vector<std::string> uris{"exp/39.mp4", "exp/37.mp4", "exp/38.mp4", "exp/37.mp4", "exp/38.mp4"};
    // std::vector<std::string> uris{"rtsp://admin:xmrbi123@192.168.175.232:554/Streaming/Channels/101"};
    // for (int i = 0; i < 64; ++i)
    // {
    //     if (i % 3 == 0)
    //         uris.emplace_back("exp/dog.mp4");
    //     else if (i % 3 == 1)
    //         uris.emplace_back("exp/cat.mp4");
    //     else if (i % 3 == 2)
    //         uris.emplace_back("exp/pig.mp4");
    // }

    auto pipeline = Pipeline::create_pipeline(det_name, pose_name, gcn_name);
    std::vector<std::string> current_uris{};

    if (pipeline == nullptr) {
        int *a;
        std::cout << "pipeline create failed" << std::endl;
        return;
    }
    //
    pipeline->set_callback(callback);
    auto out = pipeline->make_views(uris);
    for (auto x : out) {
        std::cout << x << std::endl;
    }

    pipeline->get_uris(current_uris);
    for (auto u : current_uris) {
        std::cout << u << std::endl;
    }

    pipeline->join();
}

int app_pipeline() {
    test_pipeline();
    return 0;
}