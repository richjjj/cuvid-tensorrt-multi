/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-09 09:05:46
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-09 09:29:28
 *************************************************************************************/
#include <iostream>
#include <cmath>
#include <vector>

const double THRESHOLD = 0.1;  // 阈值

struct Coordinate {
    double x;
    double y;
};

bool isStopped(const std::vector<Coordinate> &coordinates) {
    double dx = 0;
    double dy = 0;
    for (std::size_t i = 1; i < coordinates.size(); ++i) {
        dx += fabs(coordinates[i].x - coordinates[i - 1].x);
        dy += fabs(coordinates[i].y - coordinates[i - 1].y);
    }
    return dx < THRESHOLD && dy < THRESHOLD;
}

struct Line {
    double a;
    double b;
    // double c;
};

Line getLine(const Coordinate &p1, const Coordinate &p2) {
    Line line;
    line.a = p2.y - p1.y;
    line.b = p1.x - p2.x;
    // line.c = p2.x * p1.y - p1.x * p2.y;
    return line;
}

bool isIntersect(const Line &line1, const Line &line2) {
    return fabs(line1.a * line2.b - line2.a * line1.b) > 1e-6;
}

template <typename T>
struct Point {
    T x;
    T y;
};

template <typename T>
struct Segment {
    Point<T> start;
    Point<T> end;
};

template <typename T>
T crossProduct(const Point<T> &A, const Point<T> &B) {
    return A.x * B.y - A.y * B.x;
}

template <typename T>
bool isIntersect(const Segment<T> &s1, const Segment<T> &s2) {
    Point<T> A = {s1.end.x - s1.start.x, s1.end.y - s1.start.y};
    Point<T> B = {s2.end.x - s2.start.x, s2.end.y - s2.start.y};
    Point<T> C = {s2.start.x - s1.start.x, s2.start.y - s1.start.y};
    Point<T> D = {s2.end.x - s1.start.x, s2.end.y - s1.start.y};

    T c = crossProduct(C, A);
    T d = crossProduct(D, A);

    if (c * d > 0) {
        return false;
    }

    Point<T> E = {s1.end.x - s2.start.x, s1.end.y - s2.start.y};
    Point<T> F = {s1.start.x - s2.start.x, s1.start.y - s2.start.y};

    T e = crossProduct(E, B);
    T f = crossProduct(F, B);

    if (e * f > 0) {
        return false;
    }

    return true;
}

// int main() {
//     Segment<double> s1 = {{0.0, 0.0}, {1.0, 1.0}};
//     Segment<double> s2 = {{1.0, 0.0}, {0.0, 1.0}};

//     if (isIntersect(s1, s2)) {
//         std::cout << "The segments intersect." << std::endl;
//     } else {
//         std::cout << "The segments do not intersect." << std::endl;
//     }

//     return 0;
// }
