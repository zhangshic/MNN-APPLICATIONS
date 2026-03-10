
#include "nanotrack.h"
#include <memory>
#include "utils/logging.h"

std::vector<float> convert_score(const std::vector<float> &input){
    std::vector<float> exp_values(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        exp_values[i] = std::exp(input[i]);
    }
    std::vector<float> output(256);
    for (size_t i = 256; i < input.size(); ++i) {
        output[i-256] = exp_values[i] / (exp_values[i] + exp_values[i-256]);
    }
    return output;
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

inline float softmax(float x)
{
    return fast_exp(x) / (1.0f + fast_exp(x));
}


static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i * rows + j] + pad[i*rows+j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;
            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }
    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2; 
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2; 
}

void transpose_nchw_to_nhwc(const tensor_data_s& input, tensor_data_s& output, float scale, int32_t zp) {
    const int N = input.attr.dims[0];
    const int C = input.attr.dims[1];
    const int H = input.attr.dims[2];
    const int W = input.attr.dims[3];
    
    float* src = static_cast<float*>(input.data);
    float* dst = static_cast<float*>(output.data);// 使用预分配的输出内存

    std::vector<float> output_(N * H * W * C);

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    int src_idx = n*(C*H*W) + c*(H*W) + h*W + w;
                    int dst_idx = n*(H*W*C) + h*(W*C) + w*C + c;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// 构造函数
NanoTrack::NanoTrack()
{
    t_engine_ = CreateRKNNEngine();
    x_engine_ = CreateRKNNEngine();
    h_engine_ = CreateRKNNEngine();
    t_input_tensor_.data = nullptr;
    x_input_tensor_.data = nullptr;
    h_input_tensor_1.data = nullptr;
    h_input_tensor_2.data = nullptr;
}
// 析构函数
NanoTrack::~NanoTrack()
{
    if (t_input_tensor_.data != nullptr)
    {
        free(t_input_tensor_.data);
        t_input_tensor_.data = nullptr;
    }
    if (x_input_tensor_.data != nullptr)
    {
        free(x_input_tensor_.data);
        x_input_tensor_.data = nullptr;
    }
    if (h_input_tensor_1.data != nullptr)
    {
        free(h_input_tensor_1.data);
        h_input_tensor_1.data = nullptr;
    }
    if (h_input_tensor_2.data != nullptr)
    {
        free(h_input_tensor_2.data);
        h_input_tensor_2.data = nullptr;
    }

    for (auto &tensor : t_output_tensors_)
    {
        free(tensor.data);
        tensor.data = nullptr;
    }
    for (auto &tensor : x_output_tensors_)
    {
        free(tensor.data);
        tensor.data = nullptr;
    }
    for (auto &tensor : h_output_tensors_)
    {
        free(tensor.data);
        tensor.data = nullptr;
    }
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }


// 加载模型，获取输入输出属性
nn_error_e NanoTrack::LoadModel(const char *modelTName,const char *modelXName, const char *modelHName)
{
    // 加载Target模型
    auto ret = t_engine_->LoadModelFile(modelTName);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("modelTName load model file failed");
        return ret;
    }
    // get input tensor
    auto t_input_shapes = t_engine_->GetInputShapes();

    nn_tensor_attr_to_cvimg_input_data(t_input_shapes[0], t_input_tensor_);
    t_input_tensor_.data = malloc(t_input_tensor_.attr.size);

    auto output_shapes = t_engine_->GetOutputShapes();
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        t_want_float_ = true;
        NN_LOG_WARNING("output tensor type is float16, want type set to float32");
    }

    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // tensor.attr.type = output_shapes[i].type;
        tensor.attr.type = t_want_float_ ? NN_TENSOR_FLOAT : NN_TENSOR_INT8;
        tensor.attr.index = i;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        t_output_tensors_.push_back(tensor);
        t_out_zps_.push_back(output_shapes[i].zp);
        t_out_scales_.push_back(output_shapes[i].scale);
    }

    // 加载original模型
    ret = x_engine_->LoadModelFile(modelXName);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("modelXName load model file failed");
        return ret;
    }
    // get input tensor
    auto x_input_shapes = x_engine_->GetInputShapes();

    nn_tensor_attr_to_cvimg_input_data(x_input_shapes[0], x_input_tensor_);
    x_input_tensor_.data = malloc(x_input_tensor_.attr.size);

    output_shapes = x_engine_->GetOutputShapes();
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        x_want_float_ = true;
        NN_LOG_WARNING("output tensor type is float16, want type set to float32");
    }

    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // tensor.attr.type = output_shapes[i].type;
        tensor.attr.type = x_want_float_ ? NN_TENSOR_FLOAT : NN_TENSOR_INT8;
        tensor.attr.index = i;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        x_output_tensors_.push_back(tensor);
        x_out_zps_.push_back(output_shapes[i].zp);
        x_out_scales_.push_back(output_shapes[i].scale);
    }

    // 加载rpn模型
    ret = h_engine_->LoadModelFile(modelHName);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("modelHName load model file failed");
        return ret;
    }

    // get input tensor
    auto h_input_shapes = h_engine_->GetInputShapes();

    nn_tensor_attr_to_cvimg_input_data_float(h_input_shapes[0], h_input_tensor_1);
    h_input_tensor_1.data = malloc(h_input_tensor_1.attr.size);

    nn_tensor_attr_to_cvimg_input_data_float(h_input_shapes[1], h_input_tensor_2);
    h_input_tensor_2.data = malloc(h_input_tensor_2.attr.size);

    output_shapes = h_engine_->GetOutputShapes();
    if (output_shapes[0].type == NN_TENSOR_FLOAT16)
    {
        h_want_float_ = true;
        NN_LOG_WARNING("output tensor type is float16, want type set to float32");
    }

    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // tensor.attr.type = output_shapes[i].type;
        tensor.attr.type = h_want_float_ ? NN_TENSOR_FLOAT : NN_TENSOR_INT8;
        tensor.attr.index = i;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        h_output_tensors_.push_back(tensor);
        h_out_zps_.push_back(output_shapes[i].zp);
        h_out_scales_.push_back(output_shapes[i].scale);
    }
    return NN_SUCCESS;
}


// 生成每一个格点的坐标 
void NanoTrack::create_window()
{
    int score_size= cfg.score_size;
    std::vector<float> hanning(score_size,0);
    this->window.resize(score_size*score_size, 0);

    for (int i = 0; i < score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
        hanning[i] = w;
    } 
    for (int i = 0; i < score_size; i++)
    {
        for (int j = 0; j < score_size; j++)
        {
            this->window[i*score_size+j] = hanning[i] * hanning[j]; 
        }
    }    
}

// 生成每一个格点的坐标 
void NanoTrack::create_grids()
{
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = cfg.score_size;   //16x16

    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {

            this->grid_to_search_x[i*sz+j] = j*cfg.total_stride;   
            this->grid_to_search_y[i*sz+j] = i*cfg.total_stride;
        }
    }
}

void NanoTrack::init(const cv::Mat &img, cv::Rect bbox)
{

    create_window(); 
    create_grids(); 

    cv::Point2f target_pos ={0.f, 0.f}; // cx, cy
    cv::Point2f target_sz = {0.f, 0.f}; //w,h

    target_pos.x = bbox.x + (bbox.width - 1) / 2; 
    target_pos.y = bbox.y + (bbox.height -1) / 2;
    target_sz.x=bbox.width;
    target_sz.y=bbox.height;
    
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = round(sqrt(wc_z * hc_z));  

    // 使用cv::mean计算平均通道值
    // 使用平均通道值填充
    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop;
    
    // z_crop = get_subwindow_tracking(img, target_pos, cfg.exemplar_size, int(s_z),avg_chans); //cv::Mat BGR order 

    z_crop = img(bbox); // 直接使用bbox区域
    cv::resize(z_crop, z_crop, cv::Size(cfg.exemplar_size, cfg.exemplar_size)); // 调整大小到cfg.exemplar_size

    cv::Mat target_ = z_crop.clone();
    memcpy(t_input_tensor_.data, target_.data, t_input_tensor_.attr.size);
    std::vector<tensor_data_s> t_inputs;
    // 将input_tensor_放入inputs中
    t_inputs.push_back(t_input_tensor_);
    // 运行模型
    t_engine_->Run(t_inputs, t_output_tensors_, t_want_float_);

    this->state.channel_ave=avg_chans;
    this->state.im_h=img.rows;
    this->state.im_w=img.cols;
    this->state.target_pos=target_pos;
    this->state.target_sz= target_sz;
}


float NanoTrack::track(const cv::Mat img)
{
    cv::Point target_pos = this->state.target_pos;
    cv::Point2f target_sz = this->state.target_sz;
    
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  
    float scale_z = cfg.exemplar_size / s_z;  

    float d_search = (cfg.instance_size - cfg.exemplar_size) / 2; 
    float pad = d_search / scale_z; 
    float s_x = s_z + 2*pad;

    cv::Mat x_crop;  
    x_crop  = get_subwindow_tracking(img, target_pos, cfg.instance_size, int(s_x),state.channel_ave);

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    float cls_score_max;
    
    this->update(x_crop, target_pos, target_sz, scale_z, cls_score_max);

    target_pos.x = std::max(0, std::min(state.im_w, target_pos.x));
    target_pos.y = std::max(0, std::min(state.im_h, target_pos.y));
    target_sz.x = float(std::max(10, std::min(state.im_w, int(target_sz.x))));
    target_sz.y = float(std::max(10, std::min(state.im_h, int(target_sz.y))));

    // 获取最大的置信度分数
    state.cls_score_max = cls_score_max;
    // 更新目标位置和大小
    state.target_pos = target_pos;
    state.target_sz = target_sz;
    
    float cx = target_pos.x;
    float cy = target_pos.y;
    float w = target_sz.x;
    float h = target_sz.y;

    state.bbox = cv::Rect (
        static_cast<int>(cx - w / 2),
        static_cast<int>(cy - h / 2),
        static_cast<int>(w),
        static_cast<int>(h)
    );
   
    return cls_score_max;
}


cv::Mat NanoTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = pos.x - c + 0.5;
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = pos.y - c + 0.5;
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));
    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;

    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
       
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, channel_ave);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));
    return im_path; 
}



void NanoTrack::update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz,  float scale_z, float &cls_score_max)
{

    cv::Mat x_crops_ = x_crops.clone();
    memcpy(x_input_tensor_.data, x_crops_.data, x_input_tensor_.attr.size);
    std::vector<tensor_data_s> x_inputs;
    // 将input_tensor_放入inputs中
    x_inputs.push_back(x_input_tensor_);
    // 运行模型
    x_engine_->Run(x_inputs, x_output_tensors_, x_want_float_);

    transpose_nchw_to_nhwc(t_output_tensors_[0], h_input_tensor_1, t_out_scales_[0], t_out_zps_[0]);
    transpose_nchw_to_nhwc(x_output_tensors_[0], h_input_tensor_2, x_out_scales_[0], x_out_zps_[0]);

    // 准备Head模型输入
    std::vector<tensor_data_s> h_inputs;
    h_inputs.push_back(h_input_tensor_1);  // 转置后的T特征
    h_inputs.push_back(h_input_tensor_2);  // 转置后的X特征

    h_engine_->Run(h_inputs, h_output_tensors_, h_want_float_);
    // 获取输出张量维度
    int batch = h_output_tensors_[0].attr.dims[0];    // 应为1
    int channels_cls = h_output_tensors_[0].attr.dims[1]; // 分类通道数 (2)
    int rows = h_output_tensors_[0].attr.dims[2];        // 高度 (16)
    int cols = h_output_tensors_[0].attr.dims[3];        // 宽度 (16)
    int channel_size = rows * cols;

    // 分类输出处理
    float* cls_score_output = (float*)h_output_tensors_[0].data;

    // std::vector<float> cls_score_vec = std::vector<float>(cls_score_output, cls_score_output + 2 * channel_size);
    // std::vector<float> cls_scores = convert_score(cls_score_vec);

    //取第二个通道的数据 (目标得分)
    std::vector<float> cls_scores;
    cls_scores.reserve(channel_size);
    float* cls_score_target = cls_score_output + channel_size; // 跳过背景通道
    for (int i = 0; i < channel_size; i++) {
        cls_scores.push_back(sigmoid(cls_score_target[i]));
    }

    // 边界框输出处理
    float* bbox_pred_output = (float*)h_output_tensors_[1].data;
    std::vector<float> pred_x1(channel_size), pred_y1(channel_size), 
                    pred_x2(channel_size), pred_y2(channel_size);

    // size penalty  
    std::vector<float> w(cols*rows, 0), h(cols*rows, 0); 
    
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            pred_x1[i*cols + j] = this->grid_to_search_x[i*cols + j] - bbox_pred_output[i*cols + j + channel_size*0];
            pred_y1[i*cols + j] = this->grid_to_search_y[i*cols + j] - bbox_pred_output[i*cols + j + channel_size*1];
            pred_x2[i*cols + j] = this->grid_to_search_x[i*cols + j] + bbox_pred_output[i*cols + j + channel_size*2];
            pred_y2[i*cols + j] = this->grid_to_search_y[i*cols + j] + bbox_pred_output[i*cols + j + channel_size*3];

            w[i*cols + j] = pred_x2[i*cols + j] - pred_x1[i*cols + j];
            h[i*rows + j] = pred_y2[i*rows + j] - pred_y1[i*cols + j];
        }
    }


    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows*cols,0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * cfg.penalty_k);
    }

    // window penalty
    std::vector<float> pscore(rows*cols,0);
    int r_max = 0, c_max = 0; 

    // 找到最佳预测
    float maxScore = 0; 
    int max_idx = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_scores[i]) * (1 - cfg.window_influence) + this->window[i] * cfg.window_influence; 
        if (pscore[i] > maxScore) 
        {
            // get max 
            maxScore = pscore[i]; 
            
            // 最终获取的max_idx和i相同
            // floor向下取整
            // r_max = std::floor(i / rows); 
            // c_max = ((float)i / rows - r_max) * rows;  
            // max_idx = r_max * cols + c_max;
            // printf("i = %d  ,r_max = %d, c_max = %d, max_idx = %d\n", i, r_max, c_max, max_idx);

            max_idx = i;
        }
    }

    // to real size
    float pred_x1_real = pred_x1[max_idx]; 
    float pred_y1_real = pred_y1[max_idx];
    float pred_x2_real = pred_x2[max_idx];
    float pred_y2_real = pred_y2[max_idx];

    // 中心点
    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    // 宽高
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;
    float diff_xs = pred_xs - (cfg.instance_size / 2);
    float diff_ys = pred_ys - (cfg.instance_size / 2);

    // 转换为实际图像坐标（需应用scale_z）
    diff_xs /= scale_z; 
    diff_ys /= scale_z;
    pred_w /= scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr = penalty[max_idx] * cls_scores[max_idx] * cfg.lr;

    // size rate
    auto res_xs = float (target_pos.x + diff_xs);
    auto res_ys = float (target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr) * target_sz.y;

    target_pos.x = res_xs;
    target_pos.y = res_ys;
    target_sz.x = res_w;
    target_sz.y = res_h;
    cls_score_max = cls_scores[max_idx];
}