//
// Created by G.F. Duivesteijn on 08/10/2018.
//

#ifndef HYDRA_GFNET_H
#define HYDRA_GFNET_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "ofMain.h"

namespace gf {
namespace dnn {

class Net {

private:
    tensorflow::Session *session;

public:

    Net();

    virtual ~Net();

    void readNet(const std::string &graph_file_name);

    tensorflow::Tensor tensorFromCvImage(const cv::Mat &image, const double scale, const cv::Size size, const int channels, const cv::Scalar mean, bool isRGB, bool crop, int ddepth);

    tensorflow::Tensor tensorFromCvImages(std::vector<cv::Mat> images, const double scale, cv::Size size, const int channels, const cv::Scalar mean, bool swapRB, bool crop, int ddepth);

    std::vector<tensorflow::Tensor> forward(tensorflow::Tensor input_tensor, const std::string &input_layer_name, const std::string &output_layer_name);

    std::vector<tensorflow::Tensor> forward(tensorflow::Tensor input_tensor,
            const std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict,
            const std::string &output_layer_name);

};

}
}

#endif //HYDRA_GFNET_H
