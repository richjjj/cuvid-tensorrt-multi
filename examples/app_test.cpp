/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-02 10:12:18
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-13 19:45:48
 *************************************************************************************/
// #include <iostream>
// #include <thread>
// #include <mutex>
// #include <condition_variable>
// #include <queue>

// using namespace std;

// class ProdCons {
// public:
//     ProdCons() {}

//     // 生产者线程函数
//     void producer() {
//         int i = 0;
//         while (true) {
//             unique_lock<mutex> lock(mtx);
//             while (queue_.size() == max_queue_size) {
//                 not_full.wait(lock);
//             }
//             queue_.push(i++);
//             cout << "Producing: " << i - 1 << endl;
//             not_empty.notify_one();
//             lock.unlock();
//         }
//     }

//     // 消费者线程函数
//     void consumer() {
//         while (true) {
//             unique_lock<mutex> lock(mtx);
//             while (queue_.empty()) {
//                 not_empty.wait(lock);
//             }
//             int item = queue_.front();
//             queue_.pop();
//             cout << "Consuming: " << item << endl;
//             not_full.notify_one();
//             lock.unlock();
//         }
//     }

// private:
//     mutex mtx;
//     condition_variable not_empty;
//     condition_variable not_full;
//     queue<int> queue_;
//     const int max_queue_size = 10;
// };

// int app_test() {
//     ProdCons pc;
//     thread producer_thread(&ProdCons::producer, &pc);
//     thread consumer_thread(&ProdCons::consumer, &pc);
//     producer_thread.join();
//     consumer_thread.join();
//     return 0;
// }

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int app_test() {
    // 读取原始图像
    Mat img = imread("exp/test1.jpg");

    // 定义多边形顶点坐标
    vector<Point> pts = {cv::Point(550, 482), cv::Point(1100, 533), cv::Point(877, 1134), cv::Point(141, 1105)};

    // 创建与原始图像大小相同的全黑图像
    Mat mask = Mat::zeros(img.size(), CV_8UC1);

    // 在全黑图像上绘制多边形区域
    fillPoly(mask, vector<vector<Point>>{pts}, Scalar(255));

    // 应用mask到原始图像上
    Mat masked;
    bitwise_and(img, img, masked, mask);

    // 定义仿射变换前后的四个点坐标
    vector<Point2f> src_pts = {cv::Point(550, 482), cv::Point(1100, 533), cv::Point(877, 1134), cv::Point(141, 1105)};
    vector<Point2f> dst_pts = {Point2f(0, 0), Point2f(640, 0), Point2f(640, 640), Point2f(0, 640)};

    // 计算仿射变换矩阵
    Mat M = getPerspectiveTransform(src_pts, dst_pts);

    // 进行仿射变换
    Mat resized;
    warpPerspective(img, resized, M, Size(640, 640));

    // 显示结果
    cv::imwrite("resized.jpg", resized);
    // imshow("resized", resized);
    // waitKey(0);
    // destroyAllWindows();

    return 0;
}
