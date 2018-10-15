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
    delete session;
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


tensorflow::Tensor gf::dnn::Net::tensorFromCvImageFast(cv::Mat image, const double scale, const cv::Size size,
        const int channels, const cv::Scalar &mean, bool swapRB, bool crop, int ddepth) {

    const auto n_elements_per_image = size.height * size.width * channels;
    tensorflow::TensorShape shape = tensorflow::TensorShape({1, size.height, size.width, channels});
    tensorflow::Tensor _tensor(tensorflow::DT_FLOAT, shape);
    auto _tensor_ptr = _tensor.flat<float>().data();
    image = preprocess(image, scale, size, mean, swapRB, ddepth);
    auto _image_ptr = (float *) image.data;
    std::memcpy(_tensor_ptr, _image_ptr, n_elements_per_image * sizeof(float));
    return _tensor;
}


tensorflow::Tensor gf::dnn::Net::tensorFromCvImagesFast(std::vector<cv::Mat> images, const double scale,
        const cv::Size size, const int channels, const cv::Scalar &mean, bool swapRB, bool crop, int ddepth) {

    const auto n_elements_per_image = size.height * size.width * channels;
    tensorflow::TensorShape _tensor_shape = tensorflow::TensorShape({static_cast<int64>(images.size()), size.height, size.width, channels});
    tensorflow::Tensor _tensor(tensorflow::DT_FLOAT, _tensor_shape);
    auto _tensor_ptr = _tensor.flat<float>().data();
    for (int i = 0; i < images.size(); i++) {
        images[i] = preprocess(images[i], scale, size, mean, swapRB, ddepth);
        auto _image_ptr = (float *) images[i].data;
        std::memcpy(_tensor_ptr + i * n_elements_per_image, _image_ptr, n_elements_per_image * sizeof(float));
    }
    return _tensor;
}


tensorflow::Tensor gf::dnn::Net::tensorFromCvImage(cv::Mat image, const double scale, const cv::Size size,
        const int channels, const cv::Scalar &mean, bool swapRB, bool crop, int ddepth) {

    std::vector<cv::Mat> images(1, image);
    return tensorFromCvImages(images, scale, size, channels, mean, swapRB, crop, ddepth);
}


tensorflow::Tensor gf::dnn::Net::tensorFromCvImages(std::vector<cv::Mat> images, const double scale, cv::Size size,
        const int channels, const cv::Scalar &mean, bool swapRB, bool crop, int ddepth) {

    for (int i = 0; i < images.size(); i++) {
        images[i] = preprocess(images[i], scale, size, mean, swapRB, ddepth);
    }
    tensorflow::TensorShape shape = tensorflow::TensorShape({static_cast<int64>(images.size()), size.height, size.width,
            channels});
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, shape);
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    int N = 0;
    for (auto &image: images) {
        const int _rows = image.rows;
        const int _cols = image.cols;
        const int _channels = image.channels();
        const float *data = (float *) image.data;
        for (int H = 0; H < _rows; ++H) {
            for (int W = 0; W < _cols; ++W) {
                for (int C = 0; C < _channels; ++C) {
                    input_tensor_mapped(N, H, W, C) = *(data + (H * _cols + W) * _channels + C);
                }
            }
        }
        N++;
    }
    return input_tensor;
}


std::vector<tensorflow::Tensor> gf::dnn::Net::forward(const tensorflow::Tensor input_tensor,

        const std::string &input_layer_name, const std::string &output_layer_name) {
    const std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict = {{input_layer_name, input_tensor}};
    return forward(input_tensor, feed_dict, output_layer_name);
}


std::vector<tensorflow::Tensor> gf::dnn::Net::forward(const tensorflow::Tensor input_tensor,
        const std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict, const std::string &output_layer_name) {

    tensorflow::Status status;
    std::vector<tensorflow::Tensor> outputs;
    status = session->Run(feed_dict, {output_layer_name}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
    return outputs;
}


cv::Mat gf::dnn::Net::preprocess(const cv::Mat &src, const double scale, const cv::Size &size,
        const cv::Scalar &mean, bool swapRB, int ddepth) {

    cv::Mat dst = src.clone();
    cv::Size _image_size = dst.size();
    if (size != _image_size) {
        resize(dst, dst, size, 0, 0, cv::INTER_LINEAR);
    }
    if (dst.depth() == CV_8U && ddepth == CV_32F)
        dst.convertTo(dst, CV_32F);
    cv::Scalar _mean = mean;
    if (swapRB) {
        std::swap(_mean[0], _mean[2]);
        cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    }
    dst -= _mean;
    dst *= scale;
    return dst;
}
