/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-09 09:05:46
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-14 10:14:29
 *************************************************************************************/
#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <queue>
namespace Intelligent {
using namespace std;
const double eps       = 1e-6;
const double THRESHOLD = 5;  // 阈值

bool isStopped(const vector<cv::Point2f>& coordinates) {
    int n = coordinates.size();
    if (n < 100)
        return false;

    float dx = 0;
    float dy = 0;

    dy = fabs(coordinates[0].y - coordinates[58].y) + fabs(coordinates[98].y - coordinates[58].y);
    dx = fabs(coordinates[0].x - coordinates[58].x) + fabs(coordinates[98].x - coordinates[58].x);
    return dx < THRESHOLD && dy < THRESHOLD;
}
bool isStopped(const deque<cv::Point2f>& coordinates) {
    vector<cv::Point2f> v(coordinates.begin(), coordinates.end());
    return isStopped(v);
}
bool isPointInPolygon(const vector<cv::Point2f>& polygon, const cv::Point2f& point) {
    return pointPolygonTest(polygon, point, false) >= 0;
}
struct Line {
    cv::Point2f p1;
    cv::Point2f p2;
};

// 判断车辆逆行
bool isRetrograde(const deque<cv::Point2f>& coordinates) {
    auto length = coordinates.size();
    if (length < 10)
        return false;
    float sum = 0;
    int count = 0;
    for (int i = length - 5; i < length; i++) {
        auto y1 = coordinates[i].y;
        auto y2 = coordinates[i + 1].y;
        sum += y2 - y1;
        count += 1;
    }
    if (sum > 5)  // 若为正则逆行
    {
        return true;
    } else
        return false;
}

// 判断车辆轨迹线段与给定线段是否相交
bool isIntersect(const Line& line1, const Line& line2) {
    // 可以用上车辆轨迹
    cv::Point2f p = line1.p1, q = line2.p1, r = line2.p2 - line2.p1, s = line1.p2 - line1.p1;
    float r_cross_s = r.cross(s);
    if (abs(r_cross_s) < 1e-8) {
        return false;
    }

    float t = (q - p).cross(s) / r_cross_s;
    float u = (q - p).cross(r) / r_cross_s;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        return true;
    }
    return false;
}
bool isIntersect(const Line& line1, const deque<cv::Point2f>& coordinates) {
    vector<cv::Point2f> v(coordinates.begin(), coordinates.end());
    if (v.size() < 10)
        return false;
    std::vector<Line> lines;
    for (int i = 0; i < coordinates.size() - 1; i++) {
        Line line;
        line.p1 = coordinates[0];
        line.p2 = coordinates[i + 1];
        lines.push_back(line);
    }

    // 判断车辆轨迹线段是否与给定线段相交
    int count = 0;
    for (int i = 0; i < lines.size() - 1; i++) {
        if (isIntersect(lines[i], line1)) {
            count += 1;
        }
    }
    if (count > 5)
        return true;
    return false;
}

// object速度
// float speedOfTrack(const deque<cv::Point2f>& coordinates) {
//     auto length = coordinates.size();
//     if (length < 5)
//         return 0;
//     float sum = 0;
//     int count = 0;
//     for (int i = 0; i < coordinates.size() - 1; i++) {
//         auto x1 = coordinates[i].x;
//         auto x2 = coordinates[i + 1].x;
//         sum += abs(x2 - x2);
//         count += 1;
//     }
//     return sum / count;
// }
float speedOfTrack(const deque<cv::Point2f>& coordinates) {
    auto length = coordinates.size();
    if (length < 10)
        return 0;
    float sum = 0;
    int count = 0;
    for (int i = length - 5; i < length; i++) {
        auto y1 = coordinates[i].y;
        auto y2 = coordinates[i + 1].y;
        sum += abs(y2 - y1);
        count += 1;
    }
    INFO("speed of person =%f", sum / count);
    return sum / count;
}

};  // namespace Intelligent