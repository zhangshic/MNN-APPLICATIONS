

#ifndef RK3588_DEMO_NANOTRACK_H
#define RK3588_DEMO_NANOTRACK_H

#include "engine/engine.h"
#include <opencv2/opencv.hpp>

struct Config{ 
    
    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.15;
    float window_influence = 0.475;
    float lr = 0.38;
    int exemplar_size=127;
    int instance_size=255;
    int total_stride=16;
    int score_size=16;
    float context_amount = 0.5;
};

struct State { 
    int im_h; 
    int im_w;  
    // 通道均值
    cv::Scalar channel_ave; 
    cv::Point target_pos; 
    cv::Point2f target_sz = {0.f, 0.f}; 
    float cls_score_max; 

    cv::Rect bbox; // 目标边界框
};

class NanoTrack
{
public:
    NanoTrack();
    ~NanoTrack();

    void init(const cv::Mat &img, cv::Rect bbox);
    float track(cv::Mat img);
    void update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz, float scale_z, float &cls_score_max);
    nn_error_e LoadModel(const char *modelTName,const char *modelXName, const char *modelHName);    
    
    // state  dynamic
    State state;
    // config static
    Config cfg; 

private:
    
    void create_grids(); 
    void create_window();  
    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;

private:
    /** T：模板 X：图像 H：头 */
    // 模板引擎
    std::shared_ptr<NNEngine> t_engine_;
    // 图像引擎
    std::shared_ptr<NNEngine> x_engine_;
    // 输出引擎
    std::shared_ptr<NNEngine> h_engine_;
    // 模板输入
    tensor_data_s t_input_tensor_;
    // 图像输入
    tensor_data_s x_input_tensor_;
    // 头部输入1
    tensor_data_s h_input_tensor_1;
    // 头部输入2
    tensor_data_s h_input_tensor_2;
    // 模板输出
    std::vector<tensor_data_s> t_output_tensors_;
    // 图像输出
    std::vector<tensor_data_s> x_output_tensors_;
    // 头部输出
    std::vector<tensor_data_s> h_output_tensors_;
    /**是否float */
    bool t_want_float_ = false;
    bool x_want_float_ = false;
    bool h_want_float_ = false;
    /**量化还原参数 */
    std::vector<int32_t> t_out_zps_;
    std::vector<int32_t> x_out_zps_;
    std::vector<int32_t> h_out_zps_;
    std::vector<float> t_out_scales_;
    std::vector<float> x_out_scales_;
    std::vector<float> h_out_scales_;
    

};

#endif // RK3588_DEMO_NANOTRACK_H
