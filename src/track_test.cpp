#include "task/nanotrack.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <video_file_path>" << endl;
        return -1;
    }

    string video_name = argv[1];

    // 加载模型
    NanoTrack tracker;
    tracker.LoadModel("weights/T_model_backbone.rknn", "weights/X_model_backbone.rknn", "weights/model_head.rknn");

    // 打开视频文件
    VideoCapture cap(video_name);
    if (!cap.isOpened()) {
        cerr << "Error: Unable to open video file " << video_name << endl;
        return -1;
    }

    // 获取视频帧率和帧尺寸
    double fps = cap.get(CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    // 创建视频写入器
    VideoWriter video_writer("result.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));

    // 读取第一帧并初始化
    Mat frame;
    cap >> frame;
    if (frame.empty()) {
        cerr << "Error: Unable to read first frame from video file " << video_name << endl;
        return -1;
    }
    // girl_dance
    Rect trackWindow(244,161,74,70); // 初始边界框
    cv::Mat rgb = frame.clone();
    cv::cvtColor(rgb, rgb, COLOR_BGR2RGB); // 转换为
    tracker.init(rgb, trackWindow);

    rectangle(frame, trackWindow, Scalar(255, 0, 0), 2); // 绘制初始边界框
    cv::imwrite("init_frame.jpg", frame); // 保存初始化帧

    // 追踪
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        rgb = frame.clone();
        cv::cvtColor(rgb, rgb, COLOR_BGR2RGB); // 转换为
        double t1 = getTickCount();
        float score = tracker.track(rgb);
        double t2 = getTickCount();
        double process_time_ms = (t2 - t1) * 1000 / getTickFrequency();
        double fps_value = getTickFrequency() / (t2 - t1);
        // cout << "每帧处理时间: " << process_time_ms << " ms, FPS: " << fps_value << endl;
        cv::putText(frame, "Score: " + std::to_string(score), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        cv::putText(frame, "FPS: " + std::to_string(fps_value), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        // 绘制边界框
        if(score > 0.99) { // 设定一个阈值
            rectangle(frame, tracker.state.bbox, Scalar(0, 255, 0), 2);
        } else {
            rectangle(frame, tracker.state.bbox, Scalar(0, 0, 255), 2); // 如果分数低于阈值，使用红色边框
        }
        
        // 写入视频
        video_writer.write(frame);

        // 显示追踪结果
        imshow("Tracking", frame);
        if (waitKey(30) == 27) { // 按下Esc键退出
            break;
        }
    }

    // 释放资源
    video_writer.release();
    cap.release();
    destroyAllWindows();

    return 0;
}

