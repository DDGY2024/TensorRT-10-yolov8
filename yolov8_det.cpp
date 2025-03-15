
#include "yolov8_det.h"
//using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;


bool IsPathExist(const std::string& path)
{
    if (_access(path.c_str(), 0) == 0) {
        return true;
    }
    return false;
}

bool IsFile(const std::string& path)
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

bool IsFolder(const std::string& path)
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

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};


YOLOV8_Det::YOLOV8_Det(const std::string& engine_file_path, const std::string imgpath) {
    YOLOV8_Det::yoloexec(engine_file_path, imgpath);
}

YOLOV8_Det::~YOLOV8_Det() {

    //cudaStreamSynchronize(stream);
    cudaFree(gpu_input);
    cudaFree(gpu_output);

    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }
    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
    //Tensorrt10版本不再需要手动释放
    //context->destroy();
    //engine->destroy();
    //runtime->destroy();
}



bool YOLOV8_Det::load_engine(const std::string& engine_file_path)
{
    //初始化引擎文件。
    std::ifstream input(engine_file_path, std::ios::binary);
    if (!input)
    {
        return false;
    }
    input.seekg(0, input.end);
    const size_t fsize = input.tellg();
    input.seekg(0, input.beg);
    std::vector<char> bytes(fsize);
    input.read(bytes.data(), fsize);


    //获取输入输出的变量名+创建日志文件
    runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(logger));
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(bytes.data(), bytes.size()), InferDeleter());
    if (!engine)
        return false;

    int nbio = engine->getNbIOTensors();
    const char* inputname = engine->getIOTensorName(0);
    std::cout << "input name :" << inputname << std::endl;
    const char* outputname = engine->getIOTensorName(engine->getNbIOTensors() - 1);
    std::cout << "output name :" << outputname << std::endl;

    //获取输入输出的维度信息
    Dims input_shape = engine->getTensorShape(inputname);
    Dims output_shape = engine->getTensorShape(outputname);
    inputDims = Dims4(input_shape.d[0], input_shape.d[1], input_shape.d[2], input_shape.d[3]);
    outputDims = Dims4(output_shape.d[0], output_shape.d[1], output_shape.d[2], output_shape.d[3]);

    //根据输入输出维度信息申请内存
    input_size = inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3];
    output_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2];
    input_buff = (float*)malloc(input_size * sizeof(float));
    output_buff = (float*)malloc(output_size * sizeof(float));
    cudaMalloc(&input_mem, input_size * sizeof(float));
    cudaMalloc(&output_mem, output_size * sizeof(float));


    return true;
}

void YOLOV8_Det::preprocess(std::string imgpath, cv::Mat& img, std::vector<float>& factors) {
    img = cv::imread(imgpath);

    //获取原始图像信息
    cv::Mat mat;
    int rh = img.rows;
    int rw = img.cols;
    int rc = img.channels();

    //图像色彩空间转换BGR2RGB
    cv::cvtColor(img, mat, cv::COLOR_BGR2RGB);
    //int maxImageLength = rw > rh ? rw : rh;

    // 计算缩放因子
    factors.emplace_back(img.rows / 640.0);
    factors.emplace_back(img.cols / 640.0);


    // 打印插入的值
    std::cout << " Inserted values in factors:" << std::endl;
    for (size_t i = 0; i < factors.size(); ++i) {
        std::cout << "Element " << i << ": " << factors[i] << std::endl;
    }

    // 执行图像缩放
    cv::Mat resizeImg;
    int length = 640;
    cv::resize(img, resizeImg, cv::Size(length, length), 0, 0, cv::INTER_LINEAR);

    // 将图像数据类型转换为浮点型，并归一化到[0,1]
    resizeImg.convertTo(resizeImg, CV_32FC3, 1 / 255.0);

    // 遍历每个通道，将所有通道的数据提取出来存储到一个连续的内存缓存区域
    for (int i = 0; i < resizeImg.channels(); ++i) {
        cv::extractChannel(resizeImg, cv::Mat(length, length, CV_32FC1, input_buff + i * length * length), i);
    }
    return;
}


void YOLOV8_Det::infer() {


    //创建cuda流
    cudaStreamCreate(&stream);

    //创建执行上下文context对象
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        return;
    }

    //设置输入输出的指针,将其传递给context对象
    context->setTensorAddress(engine->getIOTensorName(0), input_mem);
    context->setTensorAddress(engine->getIOTensorName(engine->getNbIOTensors() - 1), output_mem);

    //将输入数据拷贝到GPU
    cudaMemcpyAsync(input_mem, input_buff, input_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    //执行推理
    context->enqueueV3(stream);
    //将输出数据拷贝到CPU
    cudaMemcpyAsync(output_buff, output_mem, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

}

std::vector<Detections> YOLOV8_Det::postProcess(std::vector<float> factors)
{
    const int outputSize = outputDims.d[1];
    //float* output = static_cast<float*>(output_buff);
    cv::Mat outputs(84, 8400, CV_32F, output_buff);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    // Preprocessing output results
    std::vector<std::string> classes{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
    int rows = outputs.size[0];
    int dimensions = outputs.size[1];
    bool yolov8 = false;

    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows)
    {
        yolov8 = true;
        rows = outputs.size[1];
        dimensions = outputs.size[0];

        outputs = outputs.reshape(1, dimensions);
        cv::transpose(outputs, outputs);
    }

    float* data = (float*)outputs.data;

    // 打印插入的值
    std::cout << "postProcess values in factors:" << std::endl;
    for (size_t i = 0; i < factors.size(); ++i) {
        std::cout << "Element " << i << ": " << factors[i] << std::endl;
    }

    for (int i = 0; i < rows; ++i)
    {
        float* classes_scores = data + 4;

        cv::Mat scores(1, 80, CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        minMaxLoc(scores, 0, &maxClassScore, 0 , &class_id);

        if (maxClassScore > 0.27)
        {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];


            int left = int((x - 0.5 * w) * factors[1]);  // 使用宽度缩放因子
            int top = int((y - 0.5 * h) * factors[0]);   // 使用高度缩放因子
            int width = int(w * factors[1]);             // 宽度缩放
            int height = int(h * factors[0]);            // 高度缩放

            boxes.push_back(cv::Rect(left, top, width, height));
        }

        data += dimensions;
    }


    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, 0.27, 0.5, nms_result);

    std::vector<Detections> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detections result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
            dis(gen),
            dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }
    return detections;
}


void YOLOV8_Det::drawBoxes(cv::Mat frame, std::vector<Detections> result, bool show)
{
    std::cout << "Number of detections:" << result.size() << std::endl;
    for (int i = 0; i < result.size(); ++i)
    {
        Detections detection = result[i];

        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(frame, box, color, 2);

        // Detection box text
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    if (show)
    {
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols * scale, frame.rows * scale));
        cv::imshow("Inference", frame);

        cv::waitKey(-1);
    }
}

void YOLOV8_Det::deal_result(cv::Mat image, std::vector<Detections> result,bool show) {
    //cv::Mat frame(image.size);
    YOLOV8_Det::drawBoxes(image,result,show);
}

void YOLOV8_Det::yoloexec(const std::string& engine_file_path, std::string imgpath) {

    cv::Mat image = cv::imread(imgpath);

    load_engine(engine_file_path);

    preprocess(imgpath,image, factors);

    infer();

    std::vector<Detections> result =  postProcess(factors);

    deal_result(image,result,true);

}