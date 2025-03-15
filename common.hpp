#pragma once
//
// Created  on 3/15/25
//
#ifndef DETECT_NORMAL_COMMON_HPP
#define DETECT_NORMAL_COMMON_HPP
#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include<process.h>
#include <io.h>

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        const cudaError_t error_code = call;                                                                           \
        if (error_code != cudaSuccess) {                                                                               \
            printf("CUDA Error:\n");                                                                                   \
            printf("    File:       %s\n", __FILE__);                                                                  \
            printf("    Line:       %d\n", __LINE__);                                                                  \
            printf("    Error code: %d\n", error_code);                                                                \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
	int size = 1;
	for (int i = 0; i < dims.nbDims; i++) {
		size *= dims.d[i];
	}
	return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
	switch (dataType) {
	case nvinfer1::DataType::kFLOAT:
		return 4;
	case nvinfer1::DataType::kHALF:
		return 2;
	case nvinfer1::DataType::kINT32:
		return 4;
	case nvinfer1::DataType::kINT8:
		return 1;
	case nvinfer1::DataType::kBOOL:
		return 1;
	default:
		return 4;
	}
}

inline static float clamp(float val, float min, float max)
{
	return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string& path)
{
	if (_access(path.c_str(), 0) == 0) {
		return true;
	}
	return false;
}

inline bool IsFile(const std::string& path)
{
	if (!IsPathExist(path)) {
		printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
		return false;
	}
	struct stat buffer;
	if (stat(path.c_str(), &buffer) == 0)
	{
		return (buffer.st_mode & S_IFREG);
	}
	//return (stat(path.c_str(), &buffer) == 0 & S_IFREG);
}

inline bool IsFolder(const std::string& path)
{
	if (!IsPathExist(path)) {
		return false;
	}
	struct stat buffer;
	if (stat(path.c_str(), &buffer) == 0)
	{
		return (buffer.st_mode & S_IFDIR);
	}
	//return (stat(path.c_str(), &buffer) == 0 & S_IFDIR);
}

namespace det {
	struct Object {
		cv::Rect_<float> rect;
		int              label = 0;
		float            prob = 0.0;
	};

	//Det推理结果结构体
	struct Detection {
		float x1, y1, x2, y2;
		float conf;
		int cls_id;
	};


	struct InferDeleter
	{
		template <typename T>
		void operator()(T* obj) const
		{
			delete obj;
		}
	};
}  // namespace det
#endif  // DETECT_NORMAL_COMMON_HPP
