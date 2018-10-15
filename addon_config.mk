# All variables and this file are optional, if they are not present the PG and the
# makefiles will try to parse the correct values from the file system.
#
# Variables that specify exclusions can use % as a wildcard to specify that anything in
# that position will match. A partial path can also be specified to, for example, exclude
# a whole folder from the parsed paths from the file system
#
# Variables can be specified using = or +=
# = will clear the contents of that variable both specified from the file or the ones parsed
# from the file system
# += will add the values to the previous ones in the file or the ones parsed from the file
# system
#
# The PG can be used to detect errors in this file, just create a new project with this addon
# and the PG will write to the console the kind of error and in which line it is

meta:
	ADDON_NAME = ofxGFTensorflow
	ADDON_DESCRIPTION = openFrameworks addon for running pre-trained Tensorflow models, using the Tensorflow C++ library.
	ADDON_AUTHOR = Gilbert Francois Duivesteijn
	ADDON_TAGS = "deep learning" "machine learning" "numerical computation" "artificial intelligence" "tensorflow"
	ADDON_URL = https://github.com/gilbertfrancois/ofxGFTensorflow

common:
	# dependencies with other addons, a list of them separated by spaces
	# or use += in several lines
	ADDON_DEPENDENCIES  = ofxCv
	ADDON_DEPENDENCIES += ofxOpenCv

    ADDON_INCLUDES = src
    ADDON_INCLUDES += libs/tensorflow/include
    ADDON_INCLUDES += libs/tensorflow/include/external/nsync
    ADDON_INCLUDES += libs/google/include

	ADDON_SOURCES_EXCLUDE = libs/%

linux64:
	ADDON_LDFLAGS = ${OF_ROOT}/addons/ofxGFTensorflow/libs/lib/linux/libtensorflow_cc.so
	ADDON_LDFLAGS += -Wl,-rpath=${OF_ROOT}/addons/ofxGFTensorflow/libs/lib/linux

linux:
	ADDON_LDFLAGS = ${OF_ROOT}/addons/ofxGFTensorflow/libs/lib/linux/libtensorflow_cc.so
	ADDON_LDFLAGS += -Wl,-rpath=${OF_ROOT}/addons/ofxGFTensorflow/libs/lib/linux


linuxarmv6l:

linuxarmv7l:

msys2:

android/armeabi:

android/armeabi-v7a:

ios:

osx:
    #ADDON_LDFLAGS = -Xlinker -rpath -Xlinker @executable_path