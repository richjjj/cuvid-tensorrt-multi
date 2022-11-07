#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <vector>

namespace ObjectDetector {

struct Box {
    float left, top, right, bottom, confidence;
    int class_label;
    float pose[51];
    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
};

typedef std::vector<Box> BoxArray;
};  // namespace ObjectDetector

#endif  // OBJECT_DETECTOR_HPP