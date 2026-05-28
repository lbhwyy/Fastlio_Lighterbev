#include "fast_lio_sam_sc_qn.h"
#include <sstream>
#include <algorithm>
#include <cmath>

namespace {
double rotationAngleDeg(const Eigen::Matrix3d& rotation) {
    Eigen::AngleAxisd angle_axis(rotation);
    return std::abs(angle_axis.angle()) * 180.0 / M_PI;
}

cv::Mat toBgr(const cv::Mat& image) {
    cv::Mat out;
    if (image.empty()) {
        return out;
    }
    if (image.channels() == 1) {
        cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);
    } else {
        out = image.clone();
    }
    return out;
}

cv::Mat colorizeBev(const cv::Mat& gray, const cv::Scalar& color) {
    if (gray.empty()) {
        return cv::Mat();
    }
    cv::Mat src = gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
    }
    cv::Mat out(src.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int y = 0; y < src.rows; ++y) {
        const uchar* in = src.ptr<uchar>(y);
        cv::Vec3b* dst = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < src.cols; ++x) {
            const uchar v = in[x];
            if (v == 0) {
                continue;
            }
            dst[x][0] = static_cast<uchar>(std::min(255.0, color[0] * static_cast<double>(v) / 255.0));
            dst[x][1] = static_cast<uchar>(std::min(255.0, color[1] * static_cast<double>(v) / 255.0));
            dst[x][2] = static_cast<uchar>(std::min(255.0, color[2] * static_cast<double>(v) / 255.0));
        }
    }
    return out;
}

cv::Mat overlayBev(const cv::Mat& base, const cv::Mat& add) {
    if (base.empty()) {
        return add.clone();
    }
    cv::Mat out = base.clone();
    if (add.empty()) {
        return out;
    }
    for (int y = 0; y < out.rows; ++y) {
        cv::Vec3b* dst = out.ptr<cv::Vec3b>(y);
        const cv::Vec3b* src = add.ptr<cv::Vec3b>(y);
        for (int x = 0; x < out.cols; ++x) {
            if (src[x] == cv::Vec3b(0, 0, 0)) {
                continue;
            }
            if (dst[x] == cv::Vec3b(0, 0, 0)) {
                dst[x] = src[x];
            } else {
                dst[x][0] = static_cast<uchar>(std::min(255, static_cast<int>(dst[x][0]) + static_cast<int>(src[x][0])));
                dst[x][1] = static_cast<uchar>(std::min(255, static_cast<int>(dst[x][1]) + static_cast<int>(src[x][1])));
                dst[x][2] = static_cast<uchar>(std::min(255, static_cast<int>(dst[x][2]) + static_cast<int>(src[x][2])));
            }
        }
    }
    return out;
}

cv::Mat fitToPanel(const cv::Mat& image, const cv::Size& size) {
    cv::Mat panel(size, CV_8UC3, cv::Scalar(0, 0, 0));
    if (image.empty()) {
        return panel;
    }
    cv::Mat bgr = toBgr(image);
    const double scale = std::min(static_cast<double>(size.width) / static_cast<double>(bgr.cols),
                                  static_cast<double>(size.height) / static_cast<double>(bgr.rows));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(std::max(1, static_cast<int>(std::round(bgr.cols * scale))),
                                      std::max(1, static_cast<int>(std::round(bgr.rows * scale)))),
               0.0, 0.0, cv::INTER_AREA);
    const int x = (size.width - resized.cols) / 2;
    const int y = (size.height - resized.rows) / 2;
    resized.copyTo(panel(cv::Rect(x, y, resized.cols, resized.rows)));
    return panel;
}

void putLabel(cv::Mat& image, const std::string& text, const cv::Point& origin) {
    cv::putText(image, text, origin + cv::Point(2, 2), cv::FONT_HERSHEY_SIMPLEX,
                0.85, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX,
                0.85, cv::Scalar(235, 235, 235), 2, cv::LINE_AA);
}

cv::Mat makeLoopMatchPanel(const BEVFrame& query, const BEVFrame& matched, int query_idx, int top1_idx) {
    const cv::Mat q = colorizeBev(query.bev_img, cv::Scalar(255, 255, 255));
    const cv::Mat top1 = colorizeBev(matched.bev_img, cv::Scalar(0, 45, 255));
    cv::Mat initial_overlap = query.initial_overlap.empty() ? overlayBev(top1, q) : toBgr(query.initial_overlap);
    cv::Mat registered_overlap = query.registered_overlap.empty() ? initial_overlap.clone() : toBgr(query.registered_overlap);
    cv::Mat matches = query.img_matches.empty() ? initial_overlap.clone() : toBgr(query.img_matches);

    if (q.empty() && top1.empty() && initial_overlap.empty() && registered_overlap.empty() && matches.empty()) {
        return cv::Mat();
    }

    const cv::Size cell(320, 320);
    cv::Mat vis(cell.height * 2, cell.width * 3, CV_8UC3, cv::Scalar(0, 0, 0));

    fitToPanel(q, cell).copyTo(vis(cv::Rect(0, 0, cell.width, cell.height)));
    fitToPanel(top1, cell).copyTo(vis(cv::Rect(cell.width, 0, cell.width, cell.height)));
    fitToPanel(initial_overlap, cell).copyTo(vis(cv::Rect(cell.width * 2, 0, cell.width, cell.height)));
    fitToPanel(matches, cv::Size(cell.width * 2, cell.height)).copyTo(
        vis(cv::Rect(0, cell.height, cell.width * 2, cell.height)));
    fitToPanel(registered_overlap, cell).copyTo(vis(cv::Rect(cell.width * 2, cell.height, cell.width, cell.height)));

    putLabel(vis, "Query", cv::Point(18, 42));
    putLabel(vis, "Top1", cv::Point(cell.width + 18, 42));
    putLabel(vis, "Initial Overlap", cv::Point(cell.width * 2 + 18, 42));
    putLabel(vis, "Feature matching", cv::Point(18, cell.height + 42));
    putLabel(vis, "Overlap aft.", cv::Point(cell.width * 2 + 18, cell.height + 42));
    putLabel(vis, "Registration", cv::Point(cell.width * 2 + 18, cell.height + 82));

    std::ostringstream oss;
    oss << "q=" << query_idx << " top1=" << top1_idx;
    cv::putText(vis, oss.str(), cv::Point(cell.width * 2 + 18, cell.height * 2 - 22),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(190, 255, 190), 1, cv::LINE_AA);

    for (int x = cell.width; x < vis.cols; x += cell.width) {
        cv::line(vis, cv::Point(x, 0), cv::Point(x, vis.rows - 1), cv::Scalar(25, 25, 25), 1);
    }
    cv::line(vis, cv::Point(0, cell.height), cv::Point(vis.cols - 1, cell.height), cv::Scalar(25, 25, 25), 1);
    return vis;
}
}  // namespace

FastLioSamScQn::FastLioSamScQn(const ros::NodeHandle &n_private):
    nh_(n_private)
{
    ////// ROS params
    package_path_ = ros::package::getPath("fast_lio_sam_sc_qn");
    double loop_update_hz, vis_hz;
    
    LoopClosureConfig lc_config;
    auto &gc = lc_config.gicp_config_;
    auto &bc = lc_config.bev_config_;
    
    /* basic */
    nh_.param<std::string>("/basic/map_frame", map_frame_, "map");
    nh_.param<double>("/basic/loop_update_hz", loop_update_hz, 1.0);
    nh_.param<std::string>("/basic/dataset", dataset_name_, dataset_name_);
    nh_.param<double>("/basic/vis_hz", vis_hz, 0.5);
    nh_.param<bool>("/basic/enable", use_loop, use_loop);
    nh_.param<double>("/save_voxel_resolution", voxel_res_, 0.3);
    nh_.param<double>("/loop_matching_voxel_resolution", lc_config.voxel_res_, 0.3);

    nh_.param<double>("/basic/loop_exclude_time", lc_config.time_excludes, 50);
    nh_.param<int>("/basic/loop_exclude_keyframes", lc_config.key_excludes, 50);
    
    nh_.param<std::string>("/basic/odom", odom_topic_, odom_topic_);
    nh_.param<std::string>("/basic/cloud", pcd_topic_, pcd_topic_);
    nh_.param<double>("/stability/loop_timeout_ms", loop_timeout_ms_, 0.0);
    nh_.param<double>("/stability/loop_max_score", loop_max_score_, 0.35);
    nh_.param<double>("/stability/loop_noise_variance_min", loop_noise_variance_min_, 0.05);
    nh_.param<double>("/stability/loop_noise_variance_max", loop_noise_variance_max_, 1.0);
    nh_.param<double>("/stability/loop_huber_k", loop_huber_k_, 1.345);
    nh_.param<bool>("/stability/loop_bidirectional_check", loop_bidirectional_check_, true);
    nh_.param<double>("/stability/loop_bidirectional_max_translation_error",
                      loop_bidirectional_max_translation_error_, 0.8);
    nh_.param<double>("/stability/loop_bidirectional_max_rotation_error_deg",
                      loop_bidirectional_max_rotation_error_deg_, 1.5);
    nh_.param<bool>("/stability/loop_use_odom_pose_for_bev_init",
                    lc_config.use_odom_pose_for_loop_init_, true);
    /* keyframe */
    nh_.param<double>("/keyframe/keyframe_threshold", keyframe_thr_, 1.0);
    if (!nh_.getParam("/keyframe/num_submap_keyframes", lc_config.num_submap_keyframes_)) {
        nh_.param<int>("/keyframe/nusubmap_keyframes", lc_config.num_submap_keyframes_, 5);
    }
    nh_.param<bool>("/keyframe/enable_submap_matching", lc_config.enable_submap_matching_, false);
    /* nano (GICP config) */
    nh_.param<int>("/nano_gicp/thread_number", gc.nano_thread_number_, 0);
    nh_.param<double>("/nano_gicp/icp_score_threshold", gc.icp_score_thr_, 10.0);
    nh_.param<int>("/nano_gicp/correspondences_number", gc.nano_correspondences_number_, 15);
    nh_.param<double>("/nano_gicp/max_correspondence_distance", gc.max_corr_dist_, 2.0);
    nh_.param<int>("/nano_gicp/max_iter", gc.nano_max_iter_, 32);
    nh_.param<double>("/nano_gicp/transformation_epsilon", gc.transformation_epsilon_, 0.01);
    nh_.param<double>("/nano_gicp/euclidean_fitness_epsilon", gc.euclidean_fitness_epsilon_, 0.01);
    nh_.param<int>("/nano_gicp/ransac/max_iter", gc.nano_ransac_max_iter_, 5);
    nh_.param<double>("/nano_gicp/ransac/outlier_rejection_threshold", gc.ransac_outlier_rejection_threshold_, 1.0);
    /* LighterBEV */
    nh_.param<double>("/LighterBEV/resolution", bc.resolution, 0.4);
    nh_.param<bool>("/LighterBEV/downsample_enable", bev_downsample_enable_, true);
    nh_.param<double>("/LighterBEV/downsample_voxel_size", bev_downsample_voxel_size_, bc.resolution);
    nh_.param<bool>("/LighterBEV/save_png", bev_save_png_, false);
    nh_.param<std::string>("/LighterBEV/save_dir", bev_output_dir_, std::string(""));
    nh_.param<int>("/LighterBEV/ninliers", bc.ninliers, 20);
    nh_.param<int>("/LighterBEV/iterations", bc.iterations, 1000);
    nh_.param<double>("/LighterBEV/threshold", bc.threshold, 0.5);
    nh_.param<std::string>("/LighterBEV/model_path", bc.model_path, "");
    if (bc.model_path.empty())
    {
        bc.model_path = package_path_ + "/../third_party/LighterBEV/models/tool/pca_kitti_best.pt";
    }
    nh_.param<double>("/LighterBEV/pr_threshold", bc.pr_threshold, 0.66);
    nh_.param<double>("/LighterBEV/match_ratio_test", bc.match_ratio_test, 0.9);
    nh_.param<int>("/LighterBEV/min_matches", bc.min_matches, 20);
    
    /* results */
    nh_.param<bool>("/result/save_map_bag", save_map_bag_, false);
    nh_.param<bool>("/result/save_map_pcd", save_map_pcd_, false);
    nh_.param<bool>("/result/save_in_kitti_format", save_in_kitti_format_, false);
    nh_.param<bool>("/result/save_raw_keyframes_on_save_dir", save_raw_keyframes_on_save_dir_, true);
    nh_.param<double>("/result/aggregate_voxel_size", save_aggregate_voxel_size_, 0.4);
    nh_.param<int>("/result/aggregate_downsample_interval", save_aggregate_downsample_interval_, 20);
    nh_.param<std::string>("/result/seq_name", seq_name_, "");
    nh_.param<std::string>("/result/save_src_dst_dir", src_dst_save_dir_, "");
    if (!src_dst_save_dir_.empty() && !fs::create_directories(src_dst_save_dir_) && !fs::exists(src_dst_save_dir_))
    {
        ROS_WARN("[Loop] failed to create save_src_dst_dir: %s. Debug PCD saving will likely fail.",
                 src_dst_save_dir_.c_str());
    }
    
    
    loop_closure_.reset(new LoopClosure(lc_config));
    /* Initialization of GTSAM */
    gtsam::ISAM2Params isam_params_;
    isam_params_.relinearizeThreshold = 0.01;
    isam_params_.relinearizeSkip = 1;
    isam_handler_ = std::make_shared<gtsam::ISAM2>(isam_params_);
    /* ROS things */
    odom_path_.header.frame_id = map_frame_;
    corrected_path_.header.frame_id = map_frame_;
    /* publishers */
    odom_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/ori_odom", 10, true);
    path_pub_ = nh_.advertise<nav_msgs::Path>("/ori_path", 10, true);
    corrected_odom_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/corrected_odom", 10, true);
    corrected_path_pub_ = nh_.advertise<nav_msgs::Path>("/corrected_path", 10, true);
    corrected_pcd_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/corrected_map", 10, true);
    corrected_current_pcd_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/corrected_current_pcd", 10, true);
    loop_detection_pub_ = nh_.advertise<visualization_msgs::Marker>("/loop_detection", 10, true);
    realtime_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/pose_stamped", 10);
    debug_src_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/src", 10, true);
    debug_dst_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/dst", 10, true);
    debug_coarse_aligned_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/coarse_aligned_lighterbev", 10, true);
    debug_fine_aligned_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/fine_aligned_nano_gicp", 10, true);
    
    // lighterbev
    lighter_bev_pub_ = nh_.advertise<fast_lio_sam_sc_qn::BevFrame>("/keyFrameBEV", 10, true);
    
    loop_img_vis = nh_.advertise<sensor_msgs::Image>("/loop_matches", 10, true);
    // lighterbev
    //laser_odom_to_init // velodyne_cloud_registered
    /* subscribers */
    sub_odom_ = std::make_shared<message_filters::Subscriber<nav_msgs::Odometry>>(nh_, odom_topic_, 10);
    sub_pcd_ = std::make_shared<message_filters::Subscriber<sensor_msgs::PointCloud2>>(nh_, pcd_topic_, 10);
    sub_odom_pcd_sync_ = std::make_shared<message_filters::Synchronizer<odom_pcd_sync_pol>>(odom_pcd_sync_pol(10), *sub_odom_, *sub_pcd_);
    sub_odom_pcd_sync_->registerCallback(boost::bind(&FastLioSamScQn::odomPcdCallback, this, _1, _2));
    sub_save_flag_ = nh_.subscribe("/save_dir", 1, &FastLioSamScQn::saveFlagCallback, this);
    
    /* Timers */
    loop_timer_ = nh_.createTimer(ros::Duration(1 / loop_update_hz), &FastLioSamScQn::loopTimerFunc, this);
    vis_timer_ = nh_.createTimer(ros::Duration(1 / vis_hz), &FastLioSamScQn::visTimerFunc, this);

    /* save dir */
    if (bev_output_dir_.empty())
    {
        bev_output_dir_ = package_path_ + "/" + seq_name_ + "/bev_imgs";
    }
    bev_directory = bev_output_dir_;
    if (bev_save_png_)
    {
        if (fs::exists(bev_directory))
        {
            fs::remove_all(bev_directory);
            ROS_INFO("Remove old bev_directory: %s", bev_directory.c_str());
        }
        fs::create_directories(bev_directory);
        ROS_INFO("Create new bev_directory: %s", bev_directory.c_str());
    }
    
    ROS_INFO("Main class, starting node...");
}

void FastLioSamScQn::publishBevFrame(
    const cv::Mat& bev_cv_mat,
    const std::vector<Eigen::Vector3f>& points,   
    const std::vector<int>& indices,              
    bool correct_frame)
{
    fast_lio_sam_sc_qn::BevFrame bev_msg;
    
    // Header
    bev_msg.header.stamp = ros::Time::now();
    bev_msg.header.frame_id = "map";
    
    // Keyframe info
    bev_msg.correct_frame = correct_frame;
    if (correct_frame)
    {
        // loop detected
        bev_msg.cur_index = current_loop_idx_pair_.first;
        bev_msg.pre_index = current_loop_idx_pair_.second;
    }
    
    
    // 填充图像
    cv_bridge::CvImage cv_img;
    cv_img.header = bev_msg.header;
    cv_img.encoding = "mono8";   // 或 "bgr8"
    cv_img.image = bev_cv_mat;
    cv_img.toImageMsg(bev_msg.img);
    
    bev_msg.points.reserve(points.size());
    bev_msg.idx.reserve(indices.size());

    for (size_t i = 0; i < points.size(); i++) {
        geometry_msgs::Point32 pt;
        pt.x = points[i].x();
        pt.y = points[i].y();
        pt.z = points[i].z();
        bev_msg.points.push_back(pt);
        bev_msg.idx.push_back(indices[i]);
    }
    
    // 发布
    lighter_bev_pub_.publish(bev_msg);
}

void FastLioSamScQn::odomPcdCallback(const nav_msgs::OdometryConstPtr &odom_msg,
                                     const sensor_msgs::PointCloud2ConstPtr &pcd_msg)
{   
    std::lock_guard<std::mutex> callback_lock(odom_pcd_callback_mutex_);

    Eigen::Matrix4d last_odom_tf;
    last_odom_tf = current_frame_.pose_eig_;                              // to calculate delta
    current_frame_ = PosePcd(*odom_msg, *pcd_msg, current_keyframe_idx_); // to be checked if keyframe or not
    bool ground_view = true;
    if (dataset_name_ == "mars")
    {
        ground_view = false;

        // 加入当前帧，维护固定长度队列
        frame_deque.push_back(current_frame_);
        if (frame_deque.size() > accSize) {
            frame_deque.pop_front();  
        }

        // === 将窗口内所有帧配准到最新一帧（current_frame_）坐标系，融合成关键帧点云 ===
        pcl::PointCloud<pcl::PointXYZI>::Ptr fused_in_curr(new pcl::PointCloud<pcl::PointXYZI>());
        

        // 以最新一帧的纠正后位姿为参考系
        const Eigen::Matrix4d T_wb_i = current_frame_.pose_corrected_eig_;

        // 把窗口内每一帧点云变换到最新帧坐标系 b_i 下并累加
        for (const auto& Fk : frame_deque) {
            const Eigen::Matrix4d& T_wb_k = Fk.pose_corrected_eig_;
            Eigen::Matrix4d T_bi_bk = T_wb_i.inverse() * T_wb_k;
            pcl::PointCloud<pcl::PointXYZI> tmp;
            pcl::transformPointCloud(Fk.pcd_, tmp, T_bi_bk.cast<float>());
            *fused_in_curr += tmp;
        }
        
        current_frame_.acc_pcd = *fused_in_curr;
    }
    
    
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    {
        //// 1. realtime pose = last corrected odom * delta (last -> current)
        std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
        odom_delta_ = odom_delta_ * last_odom_tf.inverse() * current_frame_.pose_eig_;
        current_frame_.pose_corrected_eig_ = last_corrected_pose_ * odom_delta_;
        realtime_pose_pub_.publish(poseEigToPoseStamped(current_frame_.pose_corrected_eig_, map_frame_));
        broadcaster_.sendTransform(tf::StampedTransform(poseEigToROSTf(current_frame_.pose_corrected_eig_),
                                                        ros::Time::now(),
                                                        map_frame_,
                                                        "robot"));
    }
    
    corrected_current_pcd_pub_.publish(pclToPclRos(transformPcd(current_frame_.pcd_, current_frame_.pose_corrected_eig_), map_frame_));

    if (!is_initialized_) //// init only once
    {
        current_frame_.bev_frame.bev_img = getBEVImageParallelAndSave(current_frame_,
                                                                      4,
                                                                      bev_directory,
                                                                      ground_view,
                                                                      bev_downsample_enable_,
                                                                      bev_downsample_voxel_size_,
                                                                      bev_save_png_);
        if (!ground_view)
        {
            current_frame_.acc_pcd.clear();
            current_frame_.acc_pcd.points.shrink_to_fit();
        }
        current_frame_.bev_frame.header = pcd_msg->header;
        current_frame_.timestamp_ = pcd_msg->header.stamp.toSec();
        current_frame_.idx_ = current_keyframe_idx_;
        {
            std::lock_guard<std::mutex> lock(loop_closure_mutex_);
            loop_closure_->updateLighterBEV(current_frame_.bev_frame);
        }
        // others
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            keyframes_.push_back(current_frame_);
        }
        updateOdomsAndPaths(current_frame_);
        // graph
        auto variance_vector = (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished(); // rad*rad,
                                                                                                    // meter*meter
        gtsam::noiseModel::Diagonal::shared_ptr prior_noise = gtsam::noiseModel::Diagonal::Variances(variance_vector);
        gtsam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(0, poseEigToGtsamPose(current_frame_.pose_eig_), prior_noise));
        init_esti_.insert(current_keyframe_idx_, poseEigToGtsamPose(current_frame_.pose_eig_));
        current_keyframe_idx_++;
        is_initialized_ = true;
    }
    else
    {
        //// 2. check if keyframe
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        PosePcd latest_keyframe_snapshot;
        bool has_latest_keyframe = false;
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            if (!keyframes_.empty())
            {
                latest_keyframe_snapshot = keyframes_.back();
                has_latest_keyframe = true;
            }
        }
        if (has_latest_keyframe && checkIfKeyframe(current_frame_, latest_keyframe_snapshot))
        {
            //get bevimg
            {
                current_frame_.bev_frame.bev_img = getBEVImageParallelAndSave(current_frame_,
                                                                              4,
                                                                              bev_directory,
                                                                              ground_view,
                                                                              bev_downsample_enable_,
                                                                              bev_downsample_voxel_size_,
                                                                              bev_save_png_);
                if (!ground_view)
                {
                    current_frame_.acc_pcd.clear();
                    current_frame_.acc_pcd.points.shrink_to_fit();
                }
                current_frame_.bev_frame.header = pcd_msg->header;
                current_frame_.timestamp_ = pcd_msg->header.stamp.toSec();
                current_frame_.idx_ = current_keyframe_idx_;
                // 2-4. if so, update lighterbev
                {
                    std::lock_guard<std::mutex> lock(loop_closure_mutex_);
                    loop_closure_->updateLighterBEV(current_frame_.bev_frame);
                }
            }
            // 2-2. if so, save
            {
                std::lock_guard<std::mutex> lock(keyframes_mutex_);
                keyframes_.push_back(current_frame_);
            }
            // 2-3. if so, add to graph
            auto variance_vector = (gtsam::Vector(6) << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2).finished();
            gtsam::noiseModel::Diagonal::shared_ptr odom_noise = gtsam::noiseModel::Diagonal::Variances(variance_vector);
            gtsam::Pose3 pose_from;
            {
                std::lock_guard<std::mutex> lock(keyframes_mutex_);
                if (keyframes_.empty() || current_keyframe_idx_ == 0 ||
                    static_cast<size_t>(current_keyframe_idx_ - 1) >= keyframes_.size())
                {
                    ROS_WARN_THROTTLE(1.0, "[Odom] invalid keyframe state when adding odom factor, skip.");
                    return;
                }
                pose_from = poseEigToGtsamPose(keyframes_[current_keyframe_idx_ - 1].pose_corrected_eig_);
            }
            gtsam::Pose3 pose_to = poseEigToGtsamPose(current_frame_.pose_corrected_eig_);
            {
                std::lock_guard<std::mutex> lock(graph_mutex_);
                gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(current_keyframe_idx_ - 1,
                                                                    current_keyframe_idx_,
                                                                    pose_from.between(pose_to),
                                                                    odom_noise));
                init_esti_.insert(current_keyframe_idx_, pose_to);
            }
            current_keyframe_idx_++;
            

            //// 3. vis
            high_resolution_clock::time_point t3 = high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(vis_mutex_);
                updateOdomsAndPaths(current_frame_);
            }
            
            //// 4. optimize with graph
            high_resolution_clock::time_point t4 = high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(graph_mutex_);
                isam_handler_->update(gtsam_graph_, init_esti_);
                isam_handler_->update();
                if (loop_added_flag_) // https://github.com/TixiaoShan/LIO-SAM/issues/5#issuecomment-653752936
                {
                    isam_handler_->update();
                    isam_handler_->update();
                    isam_handler_->update();
                }
                gtsam_graph_.resize(0);
                init_esti_.clear();
            }
            
            //// 5. handle corrected results
            // get corrected poses and reset odom delta (for realtime pose pub)
            high_resolution_clock::time_point t5 = high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
                corrected_esti_ = isam_handler_->calculateEstimate();
                last_corrected_pose_ = gtsamPoseToPoseEig(corrected_esti_.at<gtsam::Pose3>(corrected_esti_.size() - 1));
                odom_delta_ = Eigen::Matrix4d::Identity();
            }
            // correct poses in keyframes
            if (loop_added_flag_)
            {   
                std::lock_guard<std::mutex> lock(keyframes_mutex_);
                const size_t n = std::min(corrected_esti_.size(), keyframes_.size());
                if (n == 0)
                {
                    loop_added_flag_ = false;
                    return;
                }
                std::vector<Eigen::Vector3f> correct_positions;
                correct_positions.resize(n);
                std::vector<int> indices;
                indices.resize(n);
                for (size_t i = 0; i < n; ++i)
                {
                    keyframes_[i].pose_corrected_eig_ = gtsamPoseToPoseEig(corrected_esti_.at<gtsam::Pose3>(i));
                    correct_positions[i] = keyframes_[i].pose_corrected_eig_.block<3, 1>(0, 3).cast<float>();
                    indices[i] = keyframes_[i].idx_;
                }
                loop_added_flag_ = false;
                //发布当前帧图像，校正之后所有关键帧位置
                publishBevFrame(current_frame_.bev_frame.bev_img,
                                correct_positions,
                                indices,
                                true); // true: correct frame
            }else{
                // 发布当前帧的位置和图像
                std::vector<Eigen::Vector3f> correct_positions = {current_frame_.pose_corrected_eig_.block<3, 1>(0, 3).cast<float>()};
                std::vector<int> indices = {current_frame_.idx_};
                publishBevFrame(current_frame_.bev_frame.bev_img,
                                correct_positions,
                                indices,
                                false); // false: not correct frame
            }
            high_resolution_clock::time_point t6 = high_resolution_clock::now();

            ROS_INFO("real: %.1f, key_add: %.1f, vis: %.1f, opt: %.1f, res: %.1f, tot: %.1fms",
                     duration_cast<microseconds>(t2 - t1).count() / 1e3,
                     duration_cast<microseconds>(t3 - t2).count() / 1e3,
                     duration_cast<microseconds>(t4 - t3).count() / 1e3,
                     duration_cast<microseconds>(t5 - t4).count() / 1e3,
                     duration_cast<microseconds>(t6 - t5).count() / 1e3,
                     duration_cast<microseconds>(t6 - t1).count() / 1e3);
        }
    }
    return;
}

void FastLioSamScQn::loopTimerFunc(const ros::TimerEvent &)
{
    if(!use_loop)
    {
        return;
    }

    if (!is_initialized_)
    {
        return;
    }
    if (!loop_closure_)
    {
        ROS_WARN_THROTTLE(2.0, "[Loop] loop_closure_ is null.");
        return;
    }
    
    PosePcd latest_keyframe;
    std::vector<PosePcd> keyframes_snapshot;
    BEVFrame candidate_bev_frame;
    Eigen::Matrix4d candidate_pose_corrected = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T4d = Eigen::Matrix4d::Identity();
    int closest_keyframe_idx = -1;
    {
        std::lock_guard<std::mutex> lock(keyframes_mutex_);
        if (keyframes_.empty())
        {
            ROS_WARN_THROTTLE(2.0, "[Loop] keyframes_ is empty.");
            return;
        }
        PosePcd &latest_keyframe_ref = keyframes_.back();
        if (latest_keyframe_ref.processed_)
        {
            return;
        }
        latest_keyframe_ref.processed_ = true;
        latest_keyframe = latest_keyframe_ref;
        keyframes_snapshot = keyframes_;
    }

    {
        std::lock_guard<std::mutex> lock(loop_closure_mutex_);
        std::pair<Eigen::Matrix4d,int> pose_pair = loop_closure_->fetchCandidateKeyframeIdx(latest_keyframe, keyframes_snapshot);
        T4d = pose_pair.first;
        closest_keyframe_idx = pose_pair.second;
    }
    if (!T4d.allFinite())
    {
        ROS_WARN_THROTTLE(2.0, "[Loop] candidate transform has NaN/Inf, skip.");
        return;
    }
    if (closest_keyframe_idx < 0 || static_cast<size_t>(closest_keyframe_idx) >= keyframes_snapshot.size())
    {
        ROS_WARN_THROTTLE(2.0, "[Loop] invalid candidate idx: %d (keyframes size: %zu).",
                          closest_keyframe_idx, keyframes_snapshot.size());
        return;
    }
    candidate_bev_frame = keyframes_snapshot[closest_keyframe_idx].bev_frame;
    candidate_pose_corrected = keyframes_snapshot[closest_keyframe_idx].pose_corrected_eig_;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    ///pub LighterBEV registration panel
    cv::Mat vis_panel = makeLoopMatchPanel(
        latest_keyframe.bev_frame,
        candidate_bev_frame,
        latest_keyframe.idx_,
        closest_keyframe_idx);
    if (vis_panel.empty()) {
        ROS_WARN("[BEV] loop visualization panel is empty, nothing to publish.");
    } else {
        std_msgs::Header hdr;
        hdr.stamp = ros::Time::now();
        hdr.frame_id = "map";
        sensor_msgs::ImagePtr msg =
            cv_bridge::CvImage(hdr, sensor_msgs::image_encodings::BGR8, vis_panel).toImageMsg();
        loop_img_vis.publish(msg);
    }
    ////pub end

    RegistrationOutput reg_output;
    {
        std::lock_guard<std::mutex> lock(loop_closure_mutex_);
        if (closest_keyframe_idx < 0 || static_cast<size_t>(closest_keyframe_idx) >= keyframes_snapshot.size())
        {
            ROS_WARN_THROTTLE(2.0, "[Loop] candidate idx out of range before registration: %d (size: %zu).",
                              closest_keyframe_idx, keyframes_snapshot.size());
            return;
        }
        reg_output = loop_closure_->performLoopClosurev2(
            latest_keyframe, keyframes_snapshot, closest_keyframe_idx, T4d);
    }
    high_resolution_clock::time_point t_reg_done = high_resolution_clock::now();
    if (loop_timeout_ms_ > 0.0)
    {
        double reg_ms = duration_cast<microseconds>(t_reg_done - t1).count() / 1e3;
        if (reg_ms > loop_timeout_ms_)
        {
            ROS_WARN("[Loop] timeout %.1f ms > %.1f ms, skip this loop.", reg_ms, loop_timeout_ms_);
            return;
        }
    }
    if (!reg_output.pose_between_eig_.allFinite())
    {
        ROS_WARN_THROTTLE(2.0, "[Loop] reg_output pose has NaN/Inf, skip.");
        return;
    }
    
    auto printReg = [](const std::string& tag,
                   const RegistrationOutput& r)
    {
        std::cout << "── " << tag << " ─────────────────────────────\n";
        std::cout << "  is_valid     : " << std::boolalpha << r.is_valid_     << '\n';
        std::cout << "  is_converged : " << std::boolalpha << r.is_converged_ << '\n';
        std::cout << "  fitness score: " << r.score_ << '\n';
        std::cout << "  pose (4×4):\n" << r.pose_between_eig_ << "\n\n";
    };
    printReg("GICP  result", reg_output);
    if (reg_output.is_valid_)
    {   
        if (!std::isfinite(reg_output.score_) || reg_output.score_ <= 0.0 || reg_output.score_ > loop_max_score_)
        {
            ROS_WARN("[Loop] rejected by score gate. score=%.6f limit=%.6f",
                     reg_output.score_, loop_max_score_);
            return;
        }
        if (loop_bidirectional_check_)
        {
            const pcl::PointCloud<PointType> src_cloud = loop_closure_->getSourceCloud();
            const pcl::PointCloud<PointType> dst_cloud = loop_closure_->getTargetCloud();
            RegistrationOutput backward_output =
                loop_closure_->GicpAlignmentwithInit(dst_cloud, src_cloud, reg_output.pose_between_eig_.inverse());
            if (!backward_output.is_converged_ || !std::isfinite(backward_output.score_) ||
                backward_output.score_ > loop_max_score_)
            {
                ROS_WARN("[Loop] rejected by bidirectional check (backward fail). fwd=%.3f bwd=%.3f",
                         reg_output.score_, backward_output.score_);
                return;
            }
            const Eigen::Matrix4d cycle = reg_output.pose_between_eig_ * backward_output.pose_between_eig_;
            const double cycle_translation = cycle.block<3, 1>(0, 3).norm();
            const double cycle_rotation_deg = rotationAngleDeg(cycle.block<3, 3>(0, 0));
            if (cycle_translation > loop_bidirectional_max_translation_error_ ||
                cycle_rotation_deg > loop_bidirectional_max_rotation_error_deg_)
            {
                ROS_WARN("[Loop] rejected by bidirectional consistency. cycle=%.3fm/%.2fdeg limit=%.3fm/%.2fdeg",
                         cycle_translation, cycle_rotation_deg,
                         loop_bidirectional_max_translation_error_,
                         loop_bidirectional_max_rotation_error_deg_);
                return;
            }
        }

        ROS_INFO("\033[1;32mLoop closure accepted. Score: %.3f\033[0m",
                 reg_output.score_);
        const auto &score = reg_output.score_;
        gtsam::Pose3 pose_from = poseEigToGtsamPose(reg_output.pose_between_eig_ * latest_keyframe.pose_corrected_eig_);
        gtsam::Pose3 pose_to = poseEigToGtsamPose(candidate_pose_corrected);
        const double clamped_variance = std::max(loop_noise_variance_min_, std::min(score, loop_noise_variance_max_));
        auto variance_vector = (gtsam::Vector(6) << clamped_variance, clamped_variance, clamped_variance,
                               clamped_variance, clamped_variance, clamped_variance).finished();
        gtsam::noiseModel::Diagonal::shared_ptr loop_base_noise = gtsam::noiseModel::Diagonal::Variances(variance_vector);
        gtsam::SharedNoiseModel loop_noise =
            gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(loop_huber_k_),
                loop_base_noise);
        {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            gtsam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(latest_keyframe.idx_,
                                                                closest_keyframe_idx,
                                                                pose_from.between(pose_to),
                                                                loop_noise));
        }
        current_loop_idx_pair_ = {latest_keyframe.idx_, closest_keyframe_idx};
        loop_idx_pairs_.push_back(current_loop_idx_pair_); // for vis
        loop_added_flag_vis_ = true;
        loop_added_flag_ = true;
    }
    else
    {
        ROS_WARN("Loop closure rejected. Score: %.3f", reg_output.score_);
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    if (!src_dst_save_dir_.empty())
    {
        if (!fs::exists(src_dst_save_dir_) && !fs::create_directories(src_dst_save_dir_))
        {
            ROS_WARN("[Loop] failed to create debug PCD directory: %s", src_dst_save_dir_.c_str());
        }
        else
        {
            const std::string src_path = src_dst_save_dir_ + "/src_" + std::to_string(latest_keyframe.idx_) + ".pcd";
            const std::string dst_path = src_dst_save_dir_ + "/dst_" + std::to_string(closest_keyframe_idx) + ".pcd";
            const int src_ret = pcl::io::savePCDFileBinary(src_path, loop_closure_->getSourceCloud());
            const int dst_ret = pcl::io::savePCDFileBinary(dst_path, loop_closure_->getTargetCloud());
            if (src_ret != 0 || dst_ret != 0)
            {
                ROS_WARN("[Loop] failed to save debug PCDs: src_ret=%d dst_ret=%d dir=%s",
                         src_ret, dst_ret, src_dst_save_dir_.c_str());
            }
        }
    }

    debug_src_pub_.publish(pclToPclRos(loop_closure_->getSourceCloud(), map_frame_));
    debug_dst_pub_.publish(pclToPclRos(loop_closure_->getTargetCloud(), map_frame_));
    debug_fine_aligned_pub_.publish(pclToPclRos(loop_closure_->getFinalAlignedCloud(), map_frame_));
    debug_coarse_aligned_pub_.publish(pclToPclRos(loop_closure_->getCoarseAlignedCloud(), map_frame_));

    ROS_INFO("loop: %.1f", duration_cast<microseconds>(t2 - t1).count() / 1e3);
    return;
}


void FastLioSamScQn::visTimerFunc(const ros::TimerEvent &)
{
    if (!is_initialized_)
    {
        return;
    }

    high_resolution_clock::time_point tv1 = high_resolution_clock::now();
    //// 1. if loop closed, correct vis data
    if (loop_added_flag_vis_)
    // copy and ready
    {
        gtsam::Values corrected_esti_copied;
        pcl::PointCloud<pcl::PointXYZ> corrected_odoms;
        nav_msgs::Path corrected_path;
        {
            std::lock_guard<std::mutex> lock(realtime_pose_mutex_);
            corrected_esti_copied = corrected_esti_;
        }
        // correct pose and path
        for (size_t i = 0; i < corrected_esti_copied.size(); ++i)
        {
            gtsam::Pose3 pose_ = corrected_esti_copied.at<gtsam::Pose3>(i);
            corrected_odoms.points.emplace_back(pose_.translation().x(), pose_.translation().y(), pose_.translation().z());
            corrected_path.poses.push_back(gtsamPoseToPoseStamped(pose_, map_frame_));
        }
        // update vis of loop constraints
        if (!loop_idx_pairs_.empty())
        {
            loop_detection_pub_.publish(getLoopMarkers(corrected_esti_copied));
        }
        // update with corrected data
        {
            std::lock_guard<std::mutex> lock(vis_mutex_);
            corrected_odoms_ = corrected_odoms;
            corrected_path_.poses = corrected_path.poses;
        }
        loop_added_flag_vis_ = false;
    }
    //// 2. publish odoms, paths
    {
        std::lock_guard<std::mutex> lock(vis_mutex_);
        odom_pub_.publish(pclToPclRos(odoms_, map_frame_));
        path_pub_.publish(odom_path_);
        corrected_odom_pub_.publish(pclToPclRos(corrected_odoms_, map_frame_));
        corrected_path_pub_.publish(corrected_path_);
    }

    //// 3. global map
    if (global_map_vis_switch_ && corrected_pcd_map_pub_.getNumSubscribers() > 0) // save time, only once
    {
        pcl::PointCloud<PointType>::Ptr corrected_map(new pcl::PointCloud<PointType>());
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            if (keyframes_.empty())
            {
                return;
            }
            corrected_map->reserve(keyframes_[0].pcd_.size() * keyframes_.size()); // it's an approximated size
            for (size_t i = 0; i < keyframes_.size(); ++i)
            {
                *corrected_map += transformPcd(keyframes_[i].pcd_, keyframes_[i].pose_corrected_eig_);
            }
        }
        const auto &voxelized_map = voxelizePcd(corrected_map, voxel_res_);
        corrected_pcd_map_pub_.publish(pclToPclRos(*voxelized_map, map_frame_));
        global_map_vis_switch_ = false;
    }
    if (!global_map_vis_switch_ && corrected_pcd_map_pub_.getNumSubscribers() == 0)
    {
        global_map_vis_switch_ = true;
    }
    high_resolution_clock::time_point tv2 = high_resolution_clock::now();
    ROS_INFO("vis: %.1fms", duration_cast<microseconds>(tv2 - tv1).count() / 1e3);
    return;
}

void FastLioSamScQn::saveRawKeyframes(const std::string &raw_keyframes_dir)
{
    if (fs::exists(raw_keyframes_dir))
    {
        fs::remove_all(raw_keyframes_dir);
    }
    fs::create_directories(raw_keyframes_dir);

    std::lock_guard<std::mutex> lock(keyframes_mutex_);
    for (size_t i = 0; i < keyframes_.size(); ++i)
    {
        std::stringstream ss;
        ss << raw_keyframes_dir << "/" << std::setw(6) << std::setfill('0') << i << ".pcd";
        pcl::io::savePCDFileBinary<PointType>(ss.str(), keyframes_[i].pcd_);
    }
    ROS_INFO("[Save] Raw keyframes saved in %s", raw_keyframes_dir.c_str());
}

pcl::PointCloud<PointType>::Ptr FastLioSamScQn::buildDownsampledAggregatedMap()
{
    pcl::PointCloud<PointType>::Ptr aggregated_map(new pcl::PointCloud<PointType>());
    const float voxel = static_cast<float>(save_aggregate_voxel_size_ > 0.0 ? save_aggregate_voxel_size_ : 0.4);
    const int ds_interval = std::max(1, save_aggregate_downsample_interval_);

    std::lock_guard<std::mutex> lock(keyframes_mutex_);
    for (size_t i = 0; i < keyframes_.size(); ++i)
    {
        pcl::PointCloud<PointType> transformed = transformPcd(keyframes_[i].pcd_, keyframes_[i].pose_corrected_eig_);
        const auto &frame_downsampled = voxelizePcd(transformed, voxel);
        *aggregated_map += *frame_downsampled;

        if ((i + 1) % ds_interval == 0)
        {
            aggregated_map = voxelizePcd(aggregated_map, voxel);
        }
    }

    if (!aggregated_map->empty())
    {
        aggregated_map = voxelizePcd(aggregated_map, voxel);
    }
    return aggregated_map;
}

void FastLioSamScQn::saveFlagCallback(const std_msgs::String::ConstPtr &msg)
{   
    std::string save_dir = msg->data != "" ? msg->data : package_path_;

    // save scans as individual pcd files and poses in KITTI format
    // Delete the scans folder if it exists and create a new one
    std::string seq_directory = save_dir + "/" + seq_name_;
    std::string scans_directory = seq_directory + "/scans";
    if (!fs::exists(seq_directory))
    {
        fs::create_directories(seq_directory);
    }
    if (save_in_kitti_format_)
    {
        ROS_INFO("\033[32;1mScans are saved in %s, following the KITTI and TUM format\033[0m", scans_directory.c_str());
        if (fs::exists(seq_directory))
        {
            fs::remove_all(seq_directory);
        }
        fs::create_directories(scans_directory);

        std::ofstream kitti_pose_file(seq_directory + "/poses_kitti.txt");
        std::ofstream tum_pose_file(seq_directory + "/poses_tum.txt");
        tum_pose_file << "#timestamp x y z qx qy qz qw\n";
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            for (size_t i = 0; i < keyframes_.size(); ++i)
            {
                // Save the point cloud
                std::stringstream ss_;
                ss_ << scans_directory << "/" << std::setw(6) << std::setfill('0') << i << ".pcd";
                ROS_INFO("Saving %s...", ss_.str().c_str());
                pcl::io::savePCDFileASCII<PointType>(ss_.str(), keyframes_[i].pcd_);

                // Save the pose in KITTI format
                const auto &pose_ = keyframes_[i].pose_corrected_eig_;
                kitti_pose_file << pose_(0, 0) << " " << pose_(0, 1) << " " << pose_(0, 2) << " "
                                << pose_(0, 3) << " " << pose_(1, 0) << " " << pose_(1, 1) << " "
                                << pose_(1, 2) << " " << pose_(1, 3) << " " << pose_(2, 0) << " "
                                << pose_(2, 1) << " " << pose_(2, 2) << " " << pose_(2, 3) << "\n";
                
                const auto &lidar_optim_pose_ = poseEigToPoseStamped(keyframes_[i].pose_corrected_eig_);
                tum_pose_file << std::fixed << std::setprecision(8) << keyframes_[i].timestamp_
                              << " " << lidar_optim_pose_.pose.position.x << " "
                              << lidar_optim_pose_.pose.position.y << " "
                              << lidar_optim_pose_.pose.position.z << " "
                              << lidar_optim_pose_.pose.orientation.x << " "
                              << lidar_optim_pose_.pose.orientation.y << " "
                              << lidar_optim_pose_.pose.orientation.z << " "
                              << lidar_optim_pose_.pose.orientation.w << "\n";
            }
        }
        kitti_pose_file.close();
        tum_pose_file.close();
        ROS_INFO("\033[32;1mScans and poses saved in .pcd and KITTI format\033[0m");
    }
    if (save_raw_keyframes_on_save_dir_)
    {
        saveRawKeyframes(seq_directory + "/keyframes_raw");
    }

    if (save_map_bag_)
    {
        rosbag::Bag bag;
        bag.open(package_path_ + "/result.bag", rosbag::bagmode::Write);
        {
            std::lock_guard<std::mutex> lock(keyframes_mutex_);
            for (size_t i = 0; i < keyframes_.size(); ++i)
            {
                ros::Time time;
                time.fromSec(keyframes_[i].timestamp_);
                bag.write("/keyframe_pcd", time, pclToPclRos(keyframes_[i].pcd_, map_frame_));
                bag.write("/keyframe_pose", time, poseEigToPoseStamped(keyframes_[i].pose_corrected_eig_));
            }
        }
        bag.close();
        ROS_INFO("\033[36;1mResult saved in .bag format!!!\033[0m");
    }

    if (save_map_pcd_)
    {
        const auto &downsampled_map = buildDownsampledAggregatedMap();
        pcl::io::savePCDFileBinary<PointType>(seq_directory + "/" + seq_name_ + "_map.pcd", *downsampled_map);
        ROS_INFO("\033[32;1mDownsampled accumulated map cloud saved in .pcd format (voxel=%.2f, interval=%d)\033[0m",
                 save_aggregate_voxel_size_,
                 save_aggregate_downsample_interval_);
    }
    
}

FastLioSamScQn::~FastLioSamScQn()
{
}

void FastLioSamScQn::updateOdomsAndPaths(const PosePcd &pose_pcd_in)
{
    odoms_.points.emplace_back(pose_pcd_in.pose_eig_(0, 3),
                               pose_pcd_in.pose_eig_(1, 3),
                               pose_pcd_in.pose_eig_(2, 3));
    corrected_odoms_.points.emplace_back(pose_pcd_in.pose_corrected_eig_(0, 3),
                                         pose_pcd_in.pose_corrected_eig_(1, 3),
                                         pose_pcd_in.pose_corrected_eig_(2, 3));
    odom_path_.poses.emplace_back(poseEigToPoseStamped(pose_pcd_in.pose_eig_, map_frame_));
    corrected_path_.poses.emplace_back(poseEigToPoseStamped(pose_pcd_in.pose_corrected_eig_, map_frame_));
    return;
}

visualization_msgs::Marker FastLioSamScQn::getLoopMarkers(const gtsam::Values &corrected_esti_in)
{
    visualization_msgs::Marker edges;
    edges.type = 5u;
    edges.scale.x = 0.12f;
    edges.header.frame_id = map_frame_;
    edges.pose.orientation.w = 1.0f;
    edges.color.r = 1.0f;
    edges.color.g = 1.0f;
    edges.color.b = 1.0f;
    edges.color.a = 1.0f;
    for (size_t i = 0; i < loop_idx_pairs_.size(); ++i)
    {
        if (loop_idx_pairs_[i].first >= corrected_esti_in.size() ||
            loop_idx_pairs_[i].second >= corrected_esti_in.size())
        {
            continue;
        }
        gtsam::Pose3 pose = corrected_esti_in.at<gtsam::Pose3>(loop_idx_pairs_[i].first);
        gtsam::Pose3 pose2 = corrected_esti_in.at<gtsam::Pose3>(loop_idx_pairs_[i].second);
        geometry_msgs::Point p, p2;
        p.x = pose.translation().x();
        p.y = pose.translation().y();
        p.z = pose.translation().z();
        p2.x = pose2.translation().x();
        p2.y = pose2.translation().y();
        p2.z = pose2.translation().z();
        edges.points.push_back(p);
        edges.points.push_back(p2);
    }
    return edges;
}

bool FastLioSamScQn::checkIfKeyframe(const PosePcd &pose_pcd_in, const PosePcd &latest_pose_pcd)
{
    return keyframe_thr_ < (latest_pose_pcd.pose_corrected_eig_.block<3, 1>(0, 3) - pose_pcd_in.pose_corrected_eig_.block<3, 1>(0, 3)).norm();
}
