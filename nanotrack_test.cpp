#include "nanotrack.h"
#include <memory>
#include "utils/logging.h"

std::vector<float> convert_score(const std::vector<float> &input){

    std::vector<float> exp_values(input.size());
    for (size_t i = 0; i < input.size(); ++i){
        exp_values[i] = std::exp(input[i]);
    }

    std::vector<float> output(256);

    for (size_t i = 256; i < input.size(); ++i){
        output[i - 256] = exp_values[i] / (exp_values[i] +exp_values[i - 256]);

    }
    return output;

}

inline float fast_exp(float x) {
    union{
        uint32_t i;
        float f;

    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

inline float softmax(float x) {
    return fast_exp(x) / (1.0f + fast_exp(x));
}

// 计算目标大小归一化
static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

//计算尺寸变化惩罚
static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h, float sz)
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
            float t = std::sqrt((w[i*rows + j] + pad[i*rows + j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;
            sz2.push_back(std::max(t,(float)1.0/t));
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

    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }
    return sz2;
}

void transpose_nchw_to_nhwx(const tensor_data_s& input, tensor_data_s& output, float scale, int32_t zp) {

    const int N = input.attr.dims[0];
    const int C = input.attr.dims[1];
    const int H = input.attr.dims[2];
    const int W = input.attr.dims[3];

    float* src = static_cast<float*>(input.data);
    float* dst = static_cast<float*>(output.data);

    std::vector<float> output_(N * H * W *C);
    for(int n = 0; n < N; ++n){
        for(int h = 0; h < H; ++h){
            for(int w = 0; w < W; ++w){
                for(int c = 0; c < C; ++c){
                    int src_idx = n*(C*H*W) + c*(H*W) + h*W + w;
                    int dst_idx = n*(H*W*C) + h*(W*C) + w*C + c;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

NanoTrack::NanoTrack()
{
    t_engine = CreateRKNNEngine();
    x_engine = CreateRKNNEngine();
    h_engine = CreateRKNNEngine();

    t_input_tensor_.data = nullptr;
    x_input_tensor_.data = nullptr;
    h_input_tensor_1.data = nullptr;
    h_input_tensor_2.data = nullptr;
}

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
        free(tensors.data);
        tensor.data = nullptr;
    }
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

nn_error_e NanoTrack::LoadModel(const char *modelTName, const char *modelXName, const char *modelHName)
{
    /////////////////////////加载T模型/////////////////////////////////
    auto ret t_engine_->LoadModelFile(modelTName);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("modelTName load model file failed");
    }

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

        tensor.attr.type = t_want_float_ ? NN_TENSOR_FLOAT : NNTENSOR_INT8;
        tensor.attr.index = i;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        t_output_tensors_.push_back(output_shapes[i].zp);
        t_out_scales_.push_back(output_shapes[i].scale);
    }

    /////////////////////////加载X模型/////////////////////////////////
    auto ret = x_engine_->LoadModelFile(modelXName);
}