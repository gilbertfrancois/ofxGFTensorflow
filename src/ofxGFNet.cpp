//
// Created by G.F. Duivesteijn on 08/10/2018.
//

#include <algorithm>
#include "ofxGFNet.h"


gf::dnn::Net::Net() {

}

gf::dnn::Net::~Net() {
    if (session != nullptr) {
        session->Close();
    }
}

void gf::dnn::Net::readNet(const std::string &graph_file_name) {
    tensorflow::GraphDef graph_def;
    tensorflow::Status status;
    status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
}


tensorflow::Tensor gf::dnn::Net::tensorFromCvImage(const cv::Mat &image, const double scale, const cv::Size size,
        const int channels, const cv::Scalar mean, bool swapRB, bool crop, int ddepth) {

    std::vector<cv::Mat> images(1, image);
    return tensorFromCvImages(images, scale, size, channels, mean, swapRB, crop, ddepth);
}


tensorflow::Tensor gf::dnn::Net::tensorFromCvImages(std::vector<cv::Mat> images, const double scale,
        cv::Size size, const int channels, const cv::Scalar mean, bool swapRB, bool crop, int ddepth) {

    CV_Assert(!images.empty());
    for (int i = 0; i < images.size(); i++) {
        cv::Size imgSize = images[i].size();
        if (size == cv::Size()) {
            size = imgSize;
        }
        if (size != imgSize) {
            resize(images[i], images[i], size, 0, 0, cv::INTER_LINEAR);
        }
        if(images[i].depth() == CV_8U && ddepth == CV_32F)
            images[i].convertTo(images[i], CV_32F);
//        if (swapRB)
//            std::swap(mean[0], mean[2]);

        images[i] -= mean;
        images[i] *= scale;
    }

    tensorflow::TensorShape shape = tensorflow::TensorShape({static_cast<int64>(images.size()), size.height, size.width,
            channels});
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, shape);
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();

    int N = 0;
    for (auto &image: images) {
        cv::Mat img_norm = image * scale;
        img_norm = img_norm - mean;

        const int _rows = img_norm.rows;
        const int _cols = img_norm.cols;
        const int _channels = img_norm.channels();
        
        const float *data = (float *) img_norm.data;
        for (int H = 0; H < _rows; ++H) {
            for (int W = 0; W < _cols; ++W) {
                for (int C = 0; C < _channels; ++C) {
                    input_tensor_mapped(N, H, W, C) = *(data + (H * _cols + W) * _channels + C);
                }
            }
        }
        N++;
    }
    ofLogVerbose("tensorFromCvImages") << input_tensor.DebugString() << std::endl;
    return input_tensor;
}

std::vector<tensorflow::Tensor> gf::dnn::Net::forward(
        const tensorflow::Tensor input_tensor,
        const std::string &input_layer_name,
        const std::string &output_layer_name) {

    const std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {{input_layer_name, input_tensor}};
    return forward(input_tensor, feed_dict, output_layer_name);
}

std::vector<tensorflow::Tensor> gf::dnn::Net::forward(
        const tensorflow::Tensor input_tensor,
        const std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict,
        const std::string &output_layer_name) {

    tensorflow::Status status;
    std::vector<tensorflow::Tensor> outputs;
    status = session->Run(feed_dict, {output_layer_name}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
    return outputs;
}
