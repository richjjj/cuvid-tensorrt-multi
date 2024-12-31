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
    if (length < 20 || length > 30)
        return false;
    float sum   = 0;
    float sum_x = 0;
    int count   = 0;
    for (int i = length - 15; i < length; i++) {
        auto y1 = coordinates[i].y;
        auto y2 = coordinates[i + 1].y;
        if (y1 < 1000 || y2 < 1000)  // 不考虑远处的
            continue;
        sum_x += (coordinates[i + 1].x - coordinates[i].x);
        sum += y2 - y1;
        count += 1;
    }
    float speed = sum / count;
    // INFO("逆行speed: %f", speed);
    if (speed > 25 & sum > 500 & sum_x < -100)  // 若为正则逆行
    {
        return false;
        // return true;
    } else
        return false;
}

// 判断车辆轨迹线段与给定线段是否相交
bool isIntersect(const Line& line1, const Line& line2) {
    // 可以用上车辆轨迹
    float min_y = std::min(line2.p1.y, line2.p2.y);
    if (line1.p1.y < min_y || line1.p2.y < min_y)
        return false;
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
// bool isIntersect(const Line& line1, const deque<cv::Point2f>& coordinates, int point_num = 5) {
//     vector<cv::Point2f> v(coordinates.begin(), coordinates.end());
//     if (v.size() < 10 || v.size() > 25)
//         return false;
//     auto last = coordinates.back();
//     if (last.y < line1.p1.y || last.y < line1.p2.y)
//         return false;
//     std::vector<Line> lines;
//     for (int i = 5; i < coordinates.size() - 1; i++) {
//         Line line;
//         line.p1 = coordinates[5];
//         line.p2 = coordinates[i + 1];
//         lines.push_back(line);
//     }

//     // 判断车辆轨迹线段是否与给定线段相交
//     int count = 0;
//     for (int i = 0; i < lines.size() - 1; i++) {
//         if (isIntersect(lines[i], line1)) {
//             count += 1;
//         }
//     }
//     if (count > point_num)
//         return true;
//     return false;
// }

// 计算向量叉积
double crossProduct(const cv::Point2f& A, const cv::Point2f& B, const cv::Point2f& P) {
    return (B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x);
}

bool isIntersect(const Line& line1, const deque<cv::Point2f>& coordinates, int offset = 0) {
    vector<cv::Point2f> v(coordinates.begin(), coordinates.end());
    if (v.size() < 5)
        return false;
    auto last = coordinates.back();
    last.x    = last.x - offset;
    if (last.y < min(line1.p1.y, line1.p2.y))
        return false;
    auto last3  = v[v.size() - 3];
    last3.x     = last3.x - offset;
    bool check1 = (crossProduct(line1.p1, line1.p2, last) > 0);
    bool check3 = (crossProduct(line1.p1, line1.p2, last3) > 0);

    auto last2  = v[v.size() - 2];
    last2.x     = last2.x - offset;
    bool check2 = (crossProduct(line1.p1, line1.p2, last2) > 0);

    return check1 ^ check2 && check1 ^ check3;
}

// 未实现的方案 如果车辆的横向位置变化超过车道宽度的一半,就认为发生了变道
bool isIntersect_v2(const Line& line1, const deque<cv::Point2f>& coordinates, float left, float top, float right,
                    float bottom, int offset = 0) {
    if (right - left > bottom - top)  // 过滤长宽比不符合的
        return false;

    vector<cv::Point2f> v(coordinates.begin(), coordinates.end());
    if (v.size() < 5 || v.size() > 100)
        return false;
    auto last = coordinates.back();
    last.x    = last.x - offset;
    if (last.y < min(line1.p1.y, line1.p2.y))
        return false;
    if (bottom > max(line1.p1.y, line1.p2.y) - 20)
        return false;

    // 判断线段左侧还是右侧
    bool flag   = false;
    auto last2  = v[v.size() - 2];
    last2.x     = last2.x - offset;
    auto check2 = crossProduct(line1.p1, line1.p2, last2);
    // if (crossProduct(line1.p1, line1.p2, last) > 0) {
    //     // 最后点在线段左侧
    //     // right bottom 则应该在右侧
    //     // 并且要交叉
    //     if (check2 < 0 && crossProduct(line1.p1, line1.p2, cv::Point2f(right, bottom)) < 0)
    //         flag = true;
    // } else {
    //     // 最后点在线段右侧
    //     // 则left, bottom 在线段左侧
    //     if (check2 > 0 && crossProduct(line1.p1, line1.p2, cv::Point2f(left, bottom)) > 0)
    //         flag = true;
    // }
    if (crossProduct(line1.p1, line1.p2, last) > 0) {
        // 最后点在线段右侧, 左手坐标系
        // 则left, bottom 在线段左侧
        // 并且要交叉
        if (check2 < 0 && crossProduct(line1.p1, line1.p2, cv::Point2f(left, bottom)) < 0 &&
            abs(last.x - last2.x) < 0.25 * (right - left))
            flag = true;
    } else {
        // 最后点在线段左侧
        // right bottom 则应该在右侧
        // 前后帧的 中心点变化不大 20241010新增
        if (check2 > 0 && crossProduct(line1.p1, line1.p2, cv::Point2f(right, bottom)) > 0 &&
            abs(last.x - last2.x) < 0.25 * (right - left))
            flag = true;
    }
    return flag;
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
    // INFO("speed of person =%f", sum / count);
    return sum / count;
}

};  // namespace Intelligent