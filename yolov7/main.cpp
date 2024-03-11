#include "config.h"
#include "model.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include <chrono>
#include <fstream>
#include <arpa/inet.h>
#include <unistd.h>

using namespace nvinfer1;

const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
static Logger gLogger;

void receiveImageFromSocket(int clientSocket, cv::Mat &image) {
    // Receive image size from Python client
    int imageSize;
    if (recv(clientSocket, &imageSize, sizeof(imageSize), 0) == -1) {
        cerr << "Error receiving image size" << endl;
        return;
    }

    // Check for a valid image size
    if (imageSize <= 0) {
        cerr << "Invalid image size: " << imageSize << endl;
        return;
    }

    cout << "Received image size: " << imageSize << " bytes" << endl;

    // Receive image data from Python client
    vector<uchar> buffer(imageSize);
    int totalReceived = 0;

    while (totalReceived < imageSize) {
        int received = recv(clientSocket, buffer.data() + totalReceived, imageSize - totalReceived, 0);

        if (received <= 0) {
            cerr << "Error receiving image data" << endl;
            return;
        }

        totalReceived += received;
    }

    cout << "Successfully received image data" << endl;

    // Decode image
    image = cv::imdecode(buffer, IMREAD_UNCHANGED);
}

void sendDetectionsToSocket(int clientSocket, std::vector<std::vector<Detection>> &detections) {
    // Serialize detections and send over the socket
    // Assuming you have a function to serialize detections to JSON or another format
    // Here, I'll assume you have such a function called serializeDetectionsToJson()
    std::string jsonDetections = serializeDetectionsToJson(detections);

    // Send the size of JSON data
    size_t size = jsonDetections.size();
    send(clientSocket, &size, sizeof(size_t), 0);

    // Send JSON data over the socket
    send(clientSocket, jsonDetections.c_str(), size, 0);
}

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);

    std::string engine_name = "";
    std::string img_dir;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " [.engine] [image_folder]" << std::endl;
        return -1;
    } else {
        engine_name = std::string(argv[1]);
        img_dir = std::string(argv[2]);
    }

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* device_buffers[2];
    float* output_buffer_host = nullptr;
    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host);

    // Create socket
    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
        cerr << "Error creating socket" << endl;
        return -1;
    }

    // Bind socket to port
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(12345);  // Choose a port number
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        cerr << "Error binding socket" << endl;
        close(serverSocket);
        return -1;
    }

    // Listen for incoming connections
    if (listen(serverSocket, 1) == -1) {
        cerr << "Error listening for connections" << endl;
        close(serverSocket);
        return -1;
    }

    cout << "Waiting for connections..." << endl;

    while (1) {
        // Accept a client connection
        int clientSocket = accept(serverSocket, NULL, NULL);
        if (clientSocket == -1) {
            cerr << "Error accepting connection" << endl;
            close(serverSocket);
            return -1;
        }

        cout << "Client connected" << endl;

        // Receive image from the client and perform inference
        cv::Mat frame;
        receiveImageFromSocket(clientSocket, frame);

        // Preprocess
        cuda_preprocess(frame, device_buffers[0], kInputW, kInputH, stream);

        // Run inference
        infer(*context, stream, (void**)device_buffers, output_buffer_host, 1);

        // NMS
        std::vector<std::vector<Detection>> res_batch;
        batch_nms(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh);

        // Send detections back to the client
        sendDetectionsToSocket(clientSocket, res_batch);

        // Close client socket
        close(clientSocket);
    }

    // Cleanup
    close(serverSocket);
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
