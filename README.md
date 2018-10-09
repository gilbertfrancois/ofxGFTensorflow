# ofxGFTensorflow



## About

This addon allows you to run Tensorflow models in your own openFrameworks projects, inspired by the OpenCV DNN module. The goal is to have a simple interface for loading a model and feeding data to the network. The addon library should work in any project, not just in a openFramework project. See remarks below how to modify it.

There is another excellent project ofxMSATensorflow by Memo Akten. Please have a look there as well and see what addon suits you best. 



## Dependencies

- ofxOpenCV
- ofxCv
- openFrameworks v0.10.0 (untested on lower versions)



## Installation

- Go to the [release](https://github.com/gilbertfrancois/ofxGFTensorflow/releases) page and download a stable version of the source _or_ type  `git clone https://github.com/gilbertfrancois/ofxGFTensorflow.git` to install this repo inside your `${openframeworks_dir}/addons` folder.
- Download the precompiled Tensorflow C++ libraries for your operating system from the [release](https://github.com/gilbertfrancois/ofxGFTensorflow/releases) page and unzip the archive with headers and libraries.
- **[Optional]** Copy the file `./libs/tensorflow/lib/{linux64 | osx}/libtensorflow_cc.so` to `/usr/local/lib` to make your programs work in runtime.
