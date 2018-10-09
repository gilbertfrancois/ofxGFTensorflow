//
// Created by G.F. Duivesteijn on 08/10/2018.
//

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


tensorflow::Tensor gf::dnn::Net::tensorFromCvImage(
        const cv::Mat &image,
        const double scale,
        const cv::Size size,
        const int channels) {
    tensorflow::TensorShape shape = tensorflow::TensorShape({static_cast<int64>(1), size.height, size.width, channels});
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, shape);
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();

    cv::Mat img_norm = image * scale;
    img_norm = img_norm - 1.0;

    const int _rows = img_norm.rows;
    const int _cols = img_norm.cols;
    const int _channels = img_norm.channels();

    const float *data = (float *) img_norm.data;
    int N = 0;
    for (int H = 0; H < _rows; ++H) {
        for (int W = 0; W < _cols; ++W) {
            for (int C = 0; C < _channels; ++C) {
                input_tensor_mapped(N, H, W, C) = *(data + (H * _cols + W) * _channels + C);
            }
        }
    }

    ofLogVerbose("tensorFromCvImage") << input_tensor.DebugString() << std::endl;
    return input_tensor;
}


tensorflow::Tensor gf::dnn::Net::tensorFromCvImages(
        const std::vector<cv::Mat> &images,
        const double scale,
        const cv::Size size,
        const int channels) {

    tensorflow::TensorShape shape = tensorflow::TensorShape({static_cast<int64>(images.size()), size.height, size.width, channels});
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, shape);
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();

    int N = 0;
    for (auto &image: images) {
        cv::Mat img_norm = image * scale;
        img_norm = img_norm - 1.0;
        
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
