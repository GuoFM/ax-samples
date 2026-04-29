/*
* AXERA is pleased to support the open source community by making ax-samples available.
*
* Copyright (c) 2026, AXERA Semiconductor Co., Ltd. All rights reserved.
*
* Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
* in compliance with the License. You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSD-3-Clause
*
* Unless required by applicable law or agreed to in writing, software distributed
* under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
* CONDITIONS OF ANY KIND, either express or implied. See the License for the
* specific language governing permissions and limitations under the License.
*/

#include <cstdio>
#include <cstring>
#include <numeric>

#include <opencv2/opencv.hpp>
#include "base/common.hpp"
#include "base/detection.hpp"
#include "middleware/io.hpp"

#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

const int DEFAULT_IMG_H = 1024;
const int DEFAULT_IMG_W = 1024;

static const char* CLASS_NAMES[] = {
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"};

static const std::vector<cv::Scalar> DOTA_COLORS = {
    cv::Scalar(255, 56, 56),   cv::Scalar(255, 159, 56),  cv::Scalar(255, 207, 56),
    cv::Scalar(180, 255, 56),  cv::Scalar(102, 255, 56),  cv::Scalar(56, 255, 122),
    cv::Scalar(56, 255, 207),  cv::Scalar(56, 207, 255),  cv::Scalar(56, 122, 255),
    cv::Scalar(102, 56, 255),  cv::Scalar(180, 56, 255),  cv::Scalar(255, 56, 207),
    cv::Scalar(255, 56, 122),  cv::Scalar(200, 200, 200), cv::Scalar(128, 128, 255),
};

int NUM_CLASS = 15;

const int DEFAULT_LOOP_COUNT = 1;

const float PROB_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
namespace ax
{
    static void get_input_data_letterbox_lefttop_rgb(const cv::Mat& mat, std::vector<uint8_t>& image,
                                                     int letterbox_rows, int letterbox_cols)
    {
        const float r = std::min(letterbox_rows * 1.f / mat.rows,
                                 letterbox_cols * 1.f / mat.cols);
        const int new_unpad_w = static_cast<int>(std::lround(r * mat.cols));
        const int new_unpad_h = static_cast<int>(std::lround(r * mat.rows));

        cv::Mat resized;
        if (mat.cols != new_unpad_w || mat.rows != new_unpad_h)
        {
            cv::resize(mat, resized, cv::Size(new_unpad_w, new_unpad_h),
                       0, 0, cv::INTER_LINEAR);
        }
        else
        {
            mat.copyTo(resized);
        }

        const float dw = (letterbox_cols - new_unpad_w) * 0.5f;
        const float dh = (letterbox_rows - new_unpad_h) * 0.5f;
        const int top    = static_cast<int>(std::lround(dh - 0.1f));
        const int bottom = static_cast<int>(std::lround(dh + 0.1f));
        const int left   = static_cast<int>(std::lround(dw - 0.1f));
        const int right  = static_cast<int>(std::lround(dw + 0.1f));

        cv::Mat img_new(letterbox_rows, letterbox_cols, CV_8UC3, image.data());
        cv::copyMakeBorder(resized, img_new, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        cv::cvtColor(img_new, img_new, cv::COLOR_BGR2RGB);
    }

    static inline void cov_xywhr(float w, float h, float t, float& a, float& b, float& c)
    {
        const float ww = w * w / 12.f;
        const float hh = h * h / 12.f;
        const float cs = std::cos(t);
        const float sn = std::sin(t);
        const float c2 = cs * cs;
        const float s2 = sn * sn;
        a = ww * c2 + hh * s2;
        b = ww * s2 + hh * c2;
        c = (ww - hh) * cs * sn;
    }

    static inline float probiou(const detection::Object& A, const detection::Object& B)
    {
        constexpr float eps = 1e-7f;
        float a1, b1, c1; cov_xywhr(A.rect.width,  A.rect.height,  A.angle, a1, b1, c1);
        float a2, b2, c2; cov_xywhr(B.rect.width,  B.rect.height,  B.angle, a2, b2, c2);
        const float dx = A.rect.x - B.rect.x;
        const float dy = A.rect.y - B.rect.y;
        const float sum_ab = (a1 + a2) * (b1 + b2) - (c1 + c2) * (c1 + c2);
        const float t1 = ((a1 + a2) * dy * dy + (b1 + b2) * dx * dx) / (sum_ab + eps) * 0.25f;
        const float t2 = ((c1 + c2) * (-dx) * dy) / (sum_ab + eps) * 0.5f;
        const float i1 = std::max(a1 * b1 - c1 * c1, 0.f);
        const float i2 = std::max(a2 * b2 - c2 * c2, 0.f);
        const float t3 = std::log(sum_ab / (4.f * std::sqrt(i1 * i2) + eps) + eps) * 0.5f;
        float bd = t1 + t2 + t3;
        if (bd < eps) bd = eps;
        if (bd > 100.f) bd = 100.f;
        const float hd = std::sqrt(1.f - std::exp(-bd) + eps);
        return 1.f - hd;
    }

    static void postprocess_yolo26_obb(std::vector<detection::Object>& proposals,
                                       std::vector<detection::Object>& objects,
                                       float nms_threshold,
                                       int letterbox_rows, int letterbox_cols,
                                       int src_rows, int src_cols,
                                       int max_det = 300,
                                       bool agnostic = false)
    {
        std::sort(proposals.begin(), proposals.end(),
                  [](const detection::Object& a, const detection::Object& b) {
                      return a.prob > b.prob;
                  });

        const int n = static_cast<int>(proposals.size());
        std::vector<int> picked;
        picked.reserve(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j)
        {
            bool suppress = false;
            for (int i = 0; i < j; ++i)
            {
                if (!agnostic && proposals[i].label != proposals[j].label)
                {
                    continue;
                }
                if (probiou(proposals[i], proposals[j]) >= nms_threshold)
                {
                    suppress = true;
                    break;
                }
            }
            if (!suppress)
            {
                picked.push_back(j);
                if (static_cast<int>(picked.size()) >= max_det)
                {
                    break;
                }
            }
        }

        const float gain = std::min(letterbox_rows * 1.f / src_rows,
                                    letterbox_cols * 1.f / src_cols);
        const float pad_x = std::round((letterbox_cols - std::round(src_cols * gain)) * 0.5f - 0.1f);
        const float pad_y = std::round((letterbox_rows - std::round(src_rows * gain)) * 0.5f - 0.1f);
        const float inv_gain = 1.f / gain;

        const float pi   = static_cast<float>(M_PI);
        const float pi_2 = static_cast<float>(M_PI_2);

        objects.resize(picked.size());
        for (size_t i = 0; i < picked.size(); ++i)
        {
            objects[i] = proposals[picked[i]];

            float xc = (objects[i].rect.x - pad_x) * inv_gain;
            float yc = (objects[i].rect.y - pad_y) * inv_gain;
            float ww = objects[i].rect.width  * inv_gain;
            float hh = objects[i].rect.height * inv_gain;

            xc = std::max(0.f, std::min(xc, static_cast<float>(src_cols)));
            yc = std::max(0.f, std::min(yc, static_cast<float>(src_rows)));

            float t = objects[i].angle;
            float t_mod_pi = std::fmod(t, pi);
            if (t_mod_pi < 0.f) t_mod_pi += pi;
            const bool swap = (t_mod_pi >= pi_2);
            float w_final = swap ? hh : ww;
            float h_final = swap ? ww : hh;
            float t_final = std::fmod(t, pi_2);
            if (t_final < 0.f) t_final += pi_2;

            objects[i].rect.x = xc;
            objects[i].rect.y = yc;
            objects[i].rect.width  = w_final;
            objects[i].rect.height = h_final;
            objects[i].angle       = t_final;
        }
    }

    static void draw_objects_yolo26_obb(const cv::Mat& bgr,
                                        const std::vector<detection::Object>& objects,
                                        const char** class_names,
                                        const std::vector<cv::Scalar>& colors,
                                        const char* output_name,
                                        int thickness = 2)
    {
        cv::Mat image = bgr.clone();
        for (size_t i = 0; i < objects.size(); ++i)
        {
            const detection::Object& obj = objects[i];
            const int label = obj.label;
            const cv::Scalar color = colors[label % static_cast<int>(colors.size())];

            float xc = obj.rect.x;
            float yc = obj.rect.y;
            float w  = obj.rect.width;
            float h  = obj.rect.height;
            float ag = obj.angle;
            float wx =  w / 2.f * std::cos(ag);
            float wy =  w / 2.f * std::sin(ag);
            float hx = -h / 2.f * std::sin(ag);
            float hy =  h / 2.f * std::cos(ag);
            cv::Point p1{(int)std::lround(xc - wx - hx), (int)std::lround(yc - wy - hy)};
            cv::Point p2{(int)std::lround(xc + wx - hx), (int)std::lround(yc + wy - hy)};
            cv::Point p3{(int)std::lround(xc + wx + hx), (int)std::lround(yc + wy + hy)};
            cv::Point p4{(int)std::lround(xc - wx + hx), (int)std::lround(yc - wy + hy)};
            std::vector<cv::Point> points = {p1, p2, p3, p4, p1};
            std::vector<std::vector<cv::Point> > contours = {points};
            cv::polylines(image, contours, true, color, thickness, cv::LINE_AA);

            char label_buf[128];
            snprintf(label_buf, sizeof(label_buf), "%s %.2f", class_names[label], obj.prob);
            int baseline = 0;
            cv::Size text_sz = cv::getTextSize(label_buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            int x_text = p1.x;
            int y_text = std::max(0, p1.y - 5);
            cv::rectangle(image,
                          cv::Point(x_text,                  y_text - text_sz.height - 2),
                          cv::Point(x_text + text_sz.width + 2, y_text + 2),
                          color, -1);
            cv::putText(image, label_buf, cv::Point(x_text + 1, y_text - 1),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            fprintf(stdout, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f, %+6.1f deg], %s\n",
                    label, obj.prob * 100, xc, yc, w, h, ag * 180.f / static_cast<float>(M_PI),
                    class_names[label]);
        }
        cv::imwrite(std::string(output_name) + ".jpg", image);
    }

    void post_process(AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data, const cv::Mat& mat, int input_w, int input_h, const std::vector<float>& time_costs)
    {
        std::vector<detection::Object> proposals;
        std::vector<detection::Object> objects;
        timer timer_postprocess;

        float* output_box_ptr[3] = {(float*)io_data->pOutputs[0].pVirAddr,
                                    (float*)io_data->pOutputs[3].pVirAddr,
                                    (float*)io_data->pOutputs[6].pVirAddr};
        float* output_cls_ptr[3] = {(float*)io_data->pOutputs[1].pVirAddr,
                                    (float*)io_data->pOutputs[4].pVirAddr,
                                    (float*)io_data->pOutputs[7].pVirAddr};
        float* output_angle_ptr[3] = {(float*)io_data->pOutputs[2].pVirAddr,
                                      (float*)io_data->pOutputs[5].pVirAddr,
                                      (float*)io_data->pOutputs[8].pVirAddr};

        for (int i = 0; i < 3; ++i)
        {
            int32_t stride = (1 << i) * 8;
            int box_ch = 4;
            int cls_ch = NUM_CLASS;
            int ang_ch = 1;
            const int box_o = i * 3;
            const int cls_o = i * 3 + 1;
            const int ang_o = i * 3 + 2;
            if (io_info != nullptr && static_cast<int>(io_info->nOutputSize) > ang_o)
            {
                const auto& bm = io_info->pOutputs[box_o];
                if (bm.nShapeSize >= 1 && bm.pShape[bm.nShapeSize - 1] > 0)
                {
                    box_ch = bm.pShape[bm.nShapeSize - 1];
                }
                const auto& cm = io_info->pOutputs[cls_o];
                if (cm.nShapeSize >= 1 && cm.pShape[cm.nShapeSize - 1] > 0)
                {
                    cls_ch = cm.pShape[cm.nShapeSize - 1];
                }
                const auto& am = io_info->pOutputs[ang_o];
                if (am.nShapeSize >= 1 && am.pShape[am.nShapeSize - 1] > 0)
                {
                    ang_ch = am.pShape[am.nShapeSize - 1];
                }
            }
            detection::obb::generate_proposals_yolo26_obb(stride, output_box_ptr[i], output_cls_ptr[i], output_angle_ptr[i],
                                                          PROB_THRESHOLD, proposals, input_w, input_h, cls_ch, box_ch, ang_ch);
        }

        postprocess_yolo26_obb(proposals, objects, NMS_THRESHOLD, input_h, input_w, mat.rows, mat.cols);
        fprintf(stdout, "post process cost time:%.2f ms \n", timer_postprocess.cost());
        fprintf(stdout, "--------------------------------------\n");
        auto total_time = std::accumulate(time_costs.begin(), time_costs.end(), 0.f);
        auto min_max_time = std::minmax_element(time_costs.begin(), time_costs.end());
        fprintf(stdout,
                "Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n",
                (int)time_costs.size(),
                total_time / (float)time_costs.size(),
                *min_max_time.second,
                *min_max_time.first);
        fprintf(stdout, "--------------------------------------\n");
        fprintf(stdout, "detection num: %zu\n", objects.size());

        draw_objects_yolo26_obb(mat, objects, CLASS_NAMES, DOTA_COLORS, "yolo26_obb_out", 2);
    }

    bool run_model(const std::string& model, const std::vector<uint8_t>& data, const int& repeat, cv::Mat& mat, int input_h, int input_w)
    {
        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        auto ret = AX_ENGINE_Init(&npu_attr);
        if (0 != ret)
        {
            return ret;
        }

        std::vector<char> model_buffer;
        if (!utilities::read_file(model, model_buffer))
        {
            fprintf(stderr, "Read Run-Joint model(%s) file failed.\n", model.c_str());
            return false;
        }

        AX_ENGINE_HANDLE handle;
        ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine creating handle is done.\n");

        ret = AX_ENGINE_CreateContext(handle);
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine creating context is done.\n");

        AX_ENGINE_IO_INFO_T* io_info;
        ret = AX_ENGINE_GetIOInfo(handle, &io_info);
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine get io info is done. \n");

        AX_ENGINE_IO_T io_data;
        ret = middleware::prepare_io(io_info, &io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine alloc io is done. \n");

        ret = middleware::push_input(data, &io_data, io_info);
        SAMPLE_AX_ENGINE_DEAL_HANDLE_IO
        fprintf(stdout, "Engine push input is done. \n");
        fprintf(stdout, "--------------------------------------\n");

        for (int i = 0; i < 5; ++i)
        {
            AX_ENGINE_RunSync(handle, &io_data);
        }

        std::vector<float> time_costs(repeat, 0);
        for (int i = 0; i < repeat; ++i)
        {
            timer tick;
            ret = AX_ENGINE_RunSync(handle, &io_data);
            time_costs[i] = tick.cost();
            SAMPLE_AX_ENGINE_DEAL_HANDLE_IO
        }

        post_process(io_info, &io_data, mat, input_w, input_h, time_costs);
        fprintf(stdout, "--------------------------------------\n");

        middleware::free_io(&io_data);
        return AX_ENGINE_DestroyHandle(handle);
    }
} // namespace ax

int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("image", 'i', "image file", true, "");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false, std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));

    cmd.add<int>("repeat", 'r', "repeat count", false, DEFAULT_LOOP_COUNT);
    cmd.parse_check(argc, argv);

    auto model_file = cmd.get<std::string>("model");
    auto image_file = cmd.get<std::string>("image");

    auto model_file_flag = utilities::file_exist(model_file);
    auto image_file_flag = utilities::file_exist(image_file);

    if (!model_file_flag | !image_file_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input file %s(%s) is not exist, please check it.\n", kind.c_str(), value.c_str());
        };

        if (!model_file_flag) { show_error("model", model_file); }
        if (!image_file_flag) { show_error("image", image_file); }

        return -1;
    }

    auto input_size_string = cmd.get<std::string>("size");

    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};

    auto input_size_flag = utilities::parse_string(input_size_string, input_size);

    if (!input_size_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input %s(%s) is not allowed, please check it.\n", kind.c_str(), value.c_str());
        };

        show_error("size", input_size_string);

        return -1;
    }

    auto repeat = cmd.get<int>("repeat");

    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "image file : %s\n", image_file.c_str());
    fprintf(stdout, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "--------------------------------------\n");

    std::vector<uint8_t> image(input_size[0] * input_size[1] * 3, 0);
    cv::Mat mat = cv::imread(image_file);
    if (mat.empty())
    {
        fprintf(stderr, "Read image failed.\n");
        return -1;
    }
    ax::get_input_data_letterbox_lefttop_rgb(mat, image, input_size[0], input_size[1]);

    AX_SYS_Init();

    {
        ax::run_model(model_file, image, repeat, mat, input_size[0], input_size[1]);
        AX_ENGINE_Deinit();
    }

    AX_SYS_Deinit();
    return 0;
}
