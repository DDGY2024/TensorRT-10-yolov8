#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define DETECT_NORMAL_YOLOV8_HPP
#include "NvInferPlugin.h"
//#include "common.hpp"
#include "fstream"
#include "NvOnnxParser.h"
#include <random>
#include <sstream>

#include "NvInfer.h"
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include<process.h>
#include <io.h>

//#define DETECT_NORMAL_YOLOV8_HPP

//#define _CRT_SECURE_NO_WARNINGS
//using namespace det;
using namespace std;
using namespace nvinfer1;

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






class Logger : public ILogger
{
	void log(Severity severity, const char* msg) noexcept override
	{
		if (severity <= Severity::kWARNING)
			std::cout << msg << std::endl;
	}
};

struct Detections
{
	int class_id{ 0 };
	std::string className{};
	float confidence{ 0.0 };
	cv::Scalar color{};
	cv::Rect box{};
};

class YOLOV8_Det {
public:

	std::vector<Detections> detections;
	YOLOV8_Det(const std::string& engine_file_path, const std::string imgpath);
	~YOLOV8_Det();
	bool load_engine(const std::string& engine_file_path);

	void preprocess(std::string imgpath, cv::Mat& img, std::vector<float>& factors);
	void infer();
	std::vector<Detections> YOLOV8_Det::postProcess(std::vector<float> factors);
	void drawBoxes(cv::Mat frame, std::vector<Detections> result, bool show);
	void deal_result(cv::Mat frame, std::vector<Detections> result, bool show);
	void yoloexec(const std::string& engine_file_path, std::string imgpath);


	std::vector<float> factors;
	std::vector<void*>   host_ptrs;
	std::vector<void*>   device_ptrs;


private:

	std::shared_ptr<nvinfer1::IRuntime> runtime = nullptr; //!< The TensorRT runtime used to deserialize the engine
	std::shared_ptr<nvinfer1::ICudaEngine> engine = nullptr;

	float* gpu_input = nullptr;
	float* gpu_output = nullptr;

	nvinfer1::Dims inputDims;  //!< The dimensions of the input to the network.
	nvinfer1::Dims outputDims;

	cudaStream_t stream = 0;

	size_t input_size;
	size_t output_size;
	Logger logger;
	std::vector<void*> bindings;
	void* input_mem{ nullptr };
	void* output_mem{ nullptr };
	float* input_buff;
	float* output_buff;



	//std::unique_ptr<nvinfer1::IExecutionContext> context;
	bool letterBoxForSquare = true;
};

int main()
{

	const std::string& engine_file_path = "E:\\del\\yolov8n.engine";
	const std::string& imgpath = "C:\\Users\\DDGY\\Downloads\\YOLOv8-main\\YOLOv8-main\\ultralytics\\assets\\bus.jpg";
	YOLOV8_Det yolov8(engine_file_path, imgpath);
	//yolov8.yoloexec();

}