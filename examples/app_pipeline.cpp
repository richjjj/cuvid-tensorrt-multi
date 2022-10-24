#include "app_decode_inference_pipeline/pipeline.hpp"
// #include "common/ilogger.hpp"
#include <iostream>
// #include <opencv2/opencv.hpp>

void callback(int callbackType,
              void *img,
              char *data,
              int datalen)
{
    // std::cout << "callbackType is " << callbackType << std::endl;
    // std::cout << "datalen is : " << datalen << std::endl;
    // std::cout << "data is: " << data << std::endl;
}
void test_pipeline()
{
    // iLogger::rmtree("imgs");
    // iLogger::mkdir("imgs");
    std::string model_name = "yolov5s_pose";
    std::vector<std::string> uris{"exp/39.mp4", "exp/37.mp4", "exp/38.mp4", "exp/37.mp4", "exp/38.mp4"};
    // for (int i = 0; i < 64; ++i)
    // {
    //     if (i % 3 == 0)
    //         uris.emplace_back("exp/dog.mp4");
    //     else if (i % 3 == 1)
    //         uris.emplace_back("exp/cat.mp4");
    //     else if (i % 3 == 2)
    //         uris.emplace_back("exp/pig.mp4");
    // }

    auto pipeline = Pipeline::create_pipeline(model_name);
    std::vector<std::string> current_uris{};

    if (pipeline == nullptr)
    {
        std::cout << "pipeline create failed" << std::endl;
        return;
    }
    //
    pipeline->set_callback(callback);
    pipeline->make_views(uris);

    pipeline->get_uris(current_uris);
    for (auto u : current_uris)
    {
        std::cout << u << std::endl;
    }

    // pipeline->join();
}

int app_pipeline()
{
    test_pipeline();
    return 0;
}