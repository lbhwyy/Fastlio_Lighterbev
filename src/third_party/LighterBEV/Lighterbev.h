#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifndef LIGHTERBEV_H
#define LIGHTERBEV_H


#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "bev_frame.h"
#include "nanoflann.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include "Lighterbev.hpp"


// LighterBEVManager 类定义
class LighterBEVManager {
public:
    // 构造函数
    LighterBEVManager(std::string model_path);
    
    // 加载模型
    bool loadmodel();
    
    // 图像转换：cv::Mat 到 torch::Tensor
    torch::Tensor cvMatToTensor(const cv::Mat& img);

    // 特征检测
    void detectBEVFeatures(BEVFrame& frame, bool save_local = false);

    // // FAST特征检测
    void getFAST(BEVFrame& frame);
    // 热更新权重
    bool reload(const std::string &state_dict_path);
    bool initialized() const { return initialized_; }
    
    // 增加一个关键帧
    void addNewKeyFrame(BEVFrame& frame);

    int detectLoopClosureIDGivenScan(const BEVFrame& frame);

    void detectBEVFeaturesBatch(BEVFrame& src, BEVFrame& dst, bool save_local);
    
    std::pair<Eigen::Matrix4d, int> poseEstimation(BEVFrame& src, BEVFrame& dst);
    
    std::pair<Eigen::Matrix3d, int> matchFeatures(BEVFrame& frame,const BEVFrame& frame_prev);
    void setMatchingParams(float ratio_thresh, int min_matches);

    void preheatModel();

    REIN model_;  // 原生 REIN 模型
    
    int   keyframes_to_exclude       = 100;     // 至少跳过最近 50 帧
    double time_to_exclude  = 50.0;    // 或者跳过 3 秒内的帧（有时间戳时生效）
    float pr_threshold        = 0.66f; // L2 距离阈值（按你模型输出调整）
    int max_iterations = 1000; // RANSAC 最大迭代次数
    double ransac_threshold = 0.5; // RANSAC 内点阈值 m
    double metric_scale_ = 0.4; // BEV pixel-to-meter scale, must match BEV resolution

private:
    // 参数变量
    std::string model_path_;  // 模型路径
    int fast_threshold_ = 10;      // FAST特征检测的阈值
    float ratio_thresh_ = 0.9f;      // 特征匹配时的比率阈值
    int min_matches_ = 20;           // 至少需要的匹配点数量
    float ransac_threshold_ = 0.5;  // RANSAC算法的阈值
    float downsample_ratio_;  // 特征下采样比例
    bool down_sample_matches_; // 是否下采样特征匹配
    bool initialized_ = false;

    std::vector<std::vector<float>> global_DB;  // 全局描述子库
    std::vector<double>             kf_times_;  // 关键帧时间戳（秒）

    // 输出（可选）
    int   last_loop_id_     = -1;
    float last_loop_dist_   = std::numeric_limits<float>::infinity();



    // OpenCV变量
    cv::Ptr<cv::FastFeatureDetector> fast_detector_;  // FAST特征检测器
    cv::Ptr<cv::BFMatcher> matcher_;  // 特征匹配器

    c10::cuda::CUDAStream stream_ = c10::cuda::CUDAStream(c10::cuda::getStreamFromPool()); 
};

#endif  // LIGHTERBEV_H
