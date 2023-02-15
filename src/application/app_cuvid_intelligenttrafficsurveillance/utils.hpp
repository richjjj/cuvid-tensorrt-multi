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
    if (n < 10)
        return false;

    float dx = 0;
    float dy = 0;
    for (std::size_t i = 1; i < n; ++i) {
        dx += fabs(coordinates[i].x - coordinates[i - 1].x);
        dy += fabs(coordinates[i].y - coordinates[i - 1].y);
    }
    return dx < THRESHOLD && dy < THRESHOLD;

    // int count = 0;
    // for (int i = 0; i < n - 1; i++) {
    //     if (velocities[i] < eps)
    //         count++;
    // }

    // double ratio = (double)count / (double)(n - 1);
    // if (ratio > 0.8)
    //     return true;
    // else
    //     return false;
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
    // 或者只判断最后两帧
    std::vector<Line> lines;
    for (int i = 0; i < coordinates.size() - 1; i++) {
        Line line;
        line.p1 = coordinates[i];
        line.p2 = coordinates[i + 1];
        lines.push_back(line);
    }

    // 判断车辆轨迹线段是否与给定线段相交
    for (int i = 0; i < lines.size() - 1; i++) {
        if (isIntersect(lines[i], line1)) {
            return true;
        }
    }
    return false;
}
};  // namespace Intelligent