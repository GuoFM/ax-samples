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

/*
* Note: For the RTMPose-M pose estimation model.
* Author: GUOFANGMING
*/

#include <cstdio>
#include <cstring>
#include <numeric>
#include <cmath>
#include <array>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "middleware/io.hpp"
#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

const int INPUT_H = 256;
const int INPUT_W = 192;
const int NUM_JOINTS = 17;
const float SIMCC_SPLIT_RATIO = 2.0f;
const float BBOX_PADDING = 1.25f;
const int DEFAULT_LOOP_COUNT = 1;

static const int COCO_SKELETON[][2] = {
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
    {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
    {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2},
    {1, 3}, {2, 4}, {3, 5}, {4, 6},
};

struct AffineInfo {
    float center[2];
    float scale[2];
};

static void get_3rd_point(const float a[2], const float b[2], float out[2])
{
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    out[0] = b[0] - dy;
    out[1] = b[1] + dx;
}

static cv::Mat get_warp_matrix(const float center[2], const float scale[2], int out_w, int out_h)
{
    float src_w = scale[0];
    float dst_w = (float)out_w;
    float dst_h = (float)out_h;

    float src_dir[2] = {0.f, src_w * -0.5f};
    float dst_dir[2] = {0.f, dst_w * -0.5f};

    float src_pts[3][2], dst_pts[3][2];
    src_pts[0][0] = center[0]; src_pts[0][1] = center[1];
    src_pts[1][0] = center[0] + src_dir[0]; src_pts[1][1] = center[1] + src_dir[1];
    get_3rd_point(src_pts[0], src_pts[1], src_pts[2]);

    dst_pts[0][0] = dst_w * 0.5f; dst_pts[0][1] = dst_h * 0.5f;
    dst_pts[1][0] = dst_w * 0.5f + dst_dir[0]; dst_pts[1][1] = dst_h * 0.5f + dst_dir[1];
    get_3rd_point(dst_pts[0], dst_pts[1], dst_pts[2]);

    cv::Point2f src_cv[3] = {
        {src_pts[0][0], src_pts[0][1]},
        {src_pts[1][0], src_pts[1][1]},
        {src_pts[2][0], src_pts[2][1]}
    };
    cv::Point2f dst_cv[3] = {
        {dst_pts[0][0], dst_pts[0][1]},
        {dst_pts[1][0], dst_pts[1][1]},
        {dst_pts[2][0], dst_pts[2][1]}
    };
    return cv::getAffineTransform(src_cv, dst_cv);
}

static AffineInfo preprocess(const cv::Mat& img_bgr, std::vector<uint8_t>& out_data)
{
    int h = img_bgr.rows, w = img_bgr.cols;
    float bbox[4] = {0.f, 0.f, (float)w, (float)h};

    AffineInfo info;
    info.center[0] = (bbox[0] + bbox[2]) * 0.5f;
    info.center[1] = (bbox[1] + bbox[3]) * 0.5f;
    info.scale[0] = (bbox[2] - bbox[0]) * BBOX_PADDING;
    info.scale[1] = (bbox[3] - bbox[1]) * BBOX_PADDING;

    float aspect = (float)INPUT_W / (float)INPUT_H;
    if (info.scale[0] > info.scale[1] * aspect) {
        info.scale[1] = info.scale[0] / aspect;
    } else {
        info.scale[0] = info.scale[1] * aspect;
    }

    cv::Mat warp_mat = get_warp_matrix(info.center, info.scale, INPUT_W, INPUT_H);
    cv::Mat warped;
    cv::warpAffine(img_bgr, warped, warp_mat, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);

    out_data.resize(INPUT_H * INPUT_W * 3);
    memcpy(out_data.data(), warped.data, out_data.size());
    return info;
}

namespace ax
{
    static float g_keypoints[NUM_JOINTS][2];
    static float g_scores[NUM_JOINTS];

    void post_process(AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data,
                      const AffineInfo& affine, const std::vector<float>& time_costs)
    {
        timer timer_postprocess;

        float* output_x = (float*)io_data->pOutputs[0].pVirAddr;
        float* output_y = (float*)io_data->pOutputs[1].pVirAddr;

        int simcc_w = io_info->pOutputs[0].pShape[2]; // 384
        int simcc_h = io_info->pOutputs[1].pShape[2]; // 512

        for (int k = 0; k < NUM_JOINTS; k++) {
            float* px = output_x + k * simcc_w;
            float* py = output_y + k * simcc_h;

            int max_x_idx = std::max_element(px, px + simcc_w) - px;
            int max_y_idx = std::max_element(py, py + simcc_h) - py;
            float max_x_val = px[max_x_idx];
            float max_y_val = py[max_y_idx];

            float kp_x = (float)max_x_idx / SIMCC_SPLIT_RATIO;
            float kp_y = (float)max_y_idx / SIMCC_SPLIT_RATIO;

            g_keypoints[k][0] = kp_x / INPUT_W * affine.scale[0] + affine.center[0] - affine.scale[0] / 2.0f;
            g_keypoints[k][1] = kp_y / INPUT_H * affine.scale[1] + affine.center[1] - affine.scale[1] / 2.0f;
            g_scores[k] = std::min(max_x_val, max_y_val);
        }

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

        int above = 0;
        for (int k = 0; k < NUM_JOINTS; k++) {
            if (g_scores[k] >= 0.3f) above++;
            fprintf(stdout, "  kp%02d: (%6.1f, %6.1f)  score=%.4f\n",
                    k, g_keypoints[k][0], g_keypoints[k][1], g_scores[k]);
        }
        fprintf(stdout, "kpts above 0.3: %d/%d\n", above, NUM_JOINTS);
    }

    void draw_and_save(cv::Mat& mat)
    {
        const float thr = 0.3f;
        for (int k = 0; k < NUM_JOINTS; k++) {
            if (g_scores[k] < thr) continue;
            cv::circle(mat, cv::Point((int)g_keypoints[k][0], (int)g_keypoints[k][1]),
                       4, cv::Scalar(0, 255, 0), -1);
        }
        for (auto& sk : COCO_SKELETON) {
            int i = sk[0], j = sk[1];
            if (g_scores[i] >= thr && g_scores[j] >= thr) {
                cv::line(mat,
                         cv::Point((int)g_keypoints[i][0], (int)g_keypoints[i][1]),
                         cv::Point((int)g_keypoints[j][0], (int)g_keypoints[j][1]),
                         cv::Scalar(255, 128, 0), 2);
            }
        }
        cv::imwrite("rtmpose_out.jpg", mat);
        fprintf(stdout, "Saved: rtmpose_out.jpg\n");
    }

    bool run_model(const std::string& model, const std::vector<uint8_t>& data,
                   const int& repeat, cv::Mat& mat, const AffineInfo& affine)
    {
        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        auto ret = AX_ENGINE_Init(&npu_attr);
        if (ret != 0) {
            fprintf(stderr, "AX_ENGINE_Init failed: 0x%x\n", ret);
            return false;
        }

        std::vector<char> model_buffer;
        if (!utilities::read_file(model, model_buffer)) {
            fprintf(stderr, "Read model file failed.\n");
            return false;
        }

        AX_ENGINE_HANDLE handle;
        ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
        if (ret != 0) {
            fprintf(stderr, "AX_ENGINE_CreateHandle failed: 0x%x\n", ret);
            return false;
        }
        fprintf(stdout, "Engine creating handle is done.\n");

        AX_ENGINE_IO_INFO_T* io_info;
        ret = AX_ENGINE_GetIOInfo(handle, &io_info);
        if (ret != 0) {
            fprintf(stderr, "AX_ENGINE_GetIOInfo failed.\n");
            AX_ENGINE_DestroyHandle(handle);
            return false;
        }
        fprintf(stdout, "Engine get io info is done.\n");

        AX_ENGINE_IO_T io_data;
        memset(&io_data, 0, sizeof(io_data));
        ret = middleware::prepare_io(io_info, &io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
        if (ret != 0) {
            fprintf(stderr, "prepare_io failed.\n");
            AX_ENGINE_DestroyHandle(handle);
            return false;
        }
        fprintf(stdout, "Engine alloc io is done.\n");

        auto& input = io_data.pInputs[0];
        memcpy(input.pVirAddr, data.data(), data.size());
        fprintf(stdout, "Engine push input is done.\n");
        fprintf(stdout, "--------------------------------------\n");

        std::vector<float> time_costs(repeat, 0);
        for (int i = 0; i < repeat; ++i) {
            timer tick;
            ret = AX_ENGINE_RunSync(handle, &io_data);
            time_costs[i] = tick.cost();
            if (ret != 0) {
                fprintf(stderr, "AX_ENGINE_RunSync failed: 0x%x\n", ret);
                break;
            }
        }

        post_process(io_info, &io_data, affine, time_costs);
        draw_and_save(mat);

        middleware::free_io(&io_data);
        AX_ENGINE_DestroyHandle(handle);
        return true;
    }
}

int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "axmodel file", true, "");
    cmd.add<std::string>("image", 'i', "image file", true, "");
    cmd.add<int>("repeat", 'r', "repeat count", false, DEFAULT_LOOP_COUNT);
    cmd.parse_check(argc, argv);

    auto model_file = cmd.get<std::string>("model");
    auto image_file = cmd.get<std::string>("image");

    if (!utilities::file_exist(model_file)) {
        fprintf(stderr, "Model file not found: %s\n", model_file.c_str());
        return -1;
    }
    if (!utilities::file_exist(image_file)) {
        fprintf(stderr, "Image file not found: %s\n", image_file.c_str());
        return -1;
    }

    auto repeat = cmd.get<int>("repeat");

    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "image file : %s\n", image_file.c_str());
    fprintf(stdout, "img_h, img_w : %d %d\n", INPUT_H, INPUT_W);
    fprintf(stdout, "--------------------------------------\n");

    cv::Mat mat = cv::imread(image_file);
    if (mat.empty()) {
        fprintf(stderr, "Read image failed: %s\n", image_file.c_str());
        return -1;
    }

    std::vector<uint8_t> image;
    AffineInfo affine = preprocess(mat, image);

    AX_SYS_Init();
    {
        ax::run_model(model_file, image, repeat, mat, affine);
        AX_ENGINE_Deinit();
    }
    AX_SYS_Deinit();
    return 0;
}
