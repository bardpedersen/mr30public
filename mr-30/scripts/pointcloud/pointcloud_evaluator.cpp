// pointcloud_evaluator.cpp

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/surface/mls.h>
#include <pcl/common/distances.h>
#include <pcl/registration/transformation_estimation.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>

struct EvaluationMetrics {
    double chamfer_distance;
    double hausdorff_distance;
    double rmse_r2g;
    double rmse_g2r;
    double f_score_1mm;    // New 1mm threshold
    double f_score_2_5mm;  // New 2.5mm threshold
    double f_score_5mm;
    double f_score_10mm;
    double recall_1mm;     // New 1mm threshold
    double recall_2_5mm;   // New 2.5mm threshold
    double recall_5mm;
    double recall_10mm;
    double precision_1mm;  // New 1mm threshold
    double precision_2_5mm; // New 2.5mm threshold
    double precision_5mm;
    double precision_10mm;
    int valid_points;
    std::vector<float> error_histogram;
};
class PointCloudEvaluator {
private:
    // Utility function to create a colormap visualization for error values
    cv::Mat colorizeErrorCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
                               const std::vector<float>& errors,
                               float max_error = 0.01) {
        // Find suitable dimensions for the image
        int image_width = 800;
        int image_height = 600;
        
        // Create a black image
        cv::Mat error_image(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));
        
        // Project 3D points to 2D image space
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        
        float scale_x = image_width * 0.9f / (max_pt.x - min_pt.x);
        float scale_y = image_height * 0.9f / (max_pt.y - min_pt.y);
        float scale = std::min(scale_x, scale_y);
        
        float offset_x = (image_width - (max_pt.x - min_pt.x) * scale) / 2;
        float offset_y = (image_height - (max_pt.y - min_pt.y) * scale) / 2;
        
        // Draw points with color based on error
        for (size_t i = 0; i < cloud->size(); i++) {
            if (!std::isfinite(cloud->points[i].z))
                continue;
                
            // Map 3D point to 2D image coordinates
            int x = static_cast<int>((cloud->points[i].x - min_pt.x) * scale + offset_x);
            int y = static_cast<int>((cloud->points[i].y - min_pt.y) * scale + offset_y);
            
            if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
                // Map error to color (JET colormap: blue-green-yellow-red)
                float normalized_error = std::min(errors[i] / max_error, 1.0f);
                cv::Vec3b color;
                
                if (normalized_error < 0.25) {
                    color[0] = 255;                                    // B
                    color[1] = static_cast<uchar>(normalized_error * 4 * 255);  // G
                    color[2] = 0;                                      // R
                } else if (normalized_error < 0.5) {
                    color[0] = static_cast<uchar>((0.5 - normalized_error) * 4 * 255);  // B
                    color[1] = 255;                                    // G
                    color[2] = static_cast<uchar>((normalized_error - 0.25) * 4 * 255); // R
                } else if (normalized_error < 0.75) {
                    color[0] = 0;                                      // B
                    color[1] = static_cast<uchar>((0.75 - normalized_error) * 4 * 255); // G
                    color[2] = 255;                                    // R
                } else {
                    color[0] = 0;                                      // B
                    color[1] = 0;                                      // G
                    color[2] = 255;                                    // R
                }
                
                // Draw a circle at the point's position
                cv::circle(error_image, cv::Point(x, y), 1, cv::Scalar(color[0], color[1], color[2]), -1);
            }
        }
        
        return error_image;
    }
    
    // Helper function to add a color scale bar to an image
    cv::Mat addColorScale(const cv::Mat& image, float min_val, float max_val) {
        // Create image with space for scale bar
        int bar_width = 30;
        int margin = 15;
        int border = bar_width + margin*3 + 60;  // width for bar + text + margins
        cv::Mat result(image.rows, image.cols + border, CV_8UC3, cv::Scalar(255, 255, 255));
        
        // Copy original image
        image.copyTo(result(cv::Rect(0, 0, image.cols, image.rows)));
        
        // Calculate colorbar position
        int bar_x = image.cols + margin;
        int bar_height = image.rows * 0.7;  // 70% of image height
        int bar_y = (image.rows - bar_height) / 2;
        
        // Draw colorbar
        for (int y = 0; y < bar_height; y++) {
            // Color from top (1.0) to bottom (0.0)
            float t = 1.0f - (float)y / bar_height;
            float value = min_val + t * (max_val - min_val);
            
            cv::Vec3b color;
            if (t < 0.25) {
                color[0] = 255;
                color[1] = static_cast<uchar>(t * 4 * 255);
                color[2] = 0;
            } else if (t < 0.5) {
                color[0] = static_cast<uchar>((0.5 - t) * 4 * 255);
                color[1] = 255;
                color[2] = static_cast<uchar>((t - 0.25) * 4 * 255);
            } else if (t < 0.75) {
                color[0] = 0;
                color[1] = static_cast<uchar>((0.75 - t) * 4 * 255);
                color[2] = 255;
            } else {
                color[0] = 0;
                color[1] = 0;
                color[2] = 255;
            }
            
            cv::rectangle(
                result, 
                cv::Point(bar_x, bar_y + y), 
                cv::Point(bar_x + bar_width, bar_y + y + 1),
                cv::Scalar(color[0], color[1], color[2]),
                -1  // filled
            );
            
            // Add labels (5 evenly spaced)
            if (y % (bar_height/4) == 0 || y == bar_height-1) {
                char label[10];
                snprintf(label, sizeof(label), "%.3f m", value);
                cv::putText(
                    result, label,
                    cv::Point(bar_x + bar_width + 5, bar_y + y + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA
                );
            }
        }
        
        // Add border and title
        cv::rectangle(result, 
                     cv::Point(bar_x, bar_y), 
                     cv::Point(bar_x + bar_width, bar_y + bar_height),
                     cv::Scalar(0, 0, 0), 1);
        
        cv::putText(result, "Error (m)", 
                   cv::Point(bar_x - 5, bar_y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        
        return result;
    }

    // New function to create six-sided error visualization
    std::vector<cv::Mat> createSixSidedErrorView(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
        const std::vector<float>& errors,
        float max_error = 0.01) { // max error in meters
        
        std::vector<cv::Mat> views(6); // Will hold all 6 projections
        
        // Define the 6 projections (which axes to use for each view)
        // Each pair represents which coordinates to use for X and Y in the projection
        std::vector<std::pair<int, int>> projections = {
            {0, 1},  // Top view (X-Y plane)
            {0, 2},  // Front view (X-Z plane)
            {1, 2},  // Side view (Y-Z plane)
            {0, 1},  // Bottom view (X-Y plane, flipped)
            {0, 2},  // Back view (X-Z plane, flipped)
            {1, 2}   // Other side view (Y-Z plane, flipped)
        };
        
        std::vector<std::string> view_names = {
            "Top", "Front", "Right", "Bottom", "Back", "Left"
        };
        
        // For each projection angle
        for (int view = 0; view < 6; view++) {
            // Image dimensions
            int image_width = 800;
            int image_height = 600;
            
            // Create a blank image
            cv::Mat error_image(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));
            
            // Get which axes to use
            int axis1 = projections[view].first;   // Primary axis (X of image)
            int axis2 = projections[view].second;  // Secondary axis (Y of image)
            bool flip_axis1 = view >= 3;           // Whether to flip the axis for opposite views
            bool flip_axis2 = view >= 3;           // Whether to flip the axis for opposite views
            
            // Find min/max values for scaling
            float min_val1 = FLT_MAX, max_val1 = -FLT_MAX;
            float min_val2 = FLT_MAX, max_val2 = -FLT_MAX;
            
            // Get min/max for the selected axes
            for (const auto& point : cloud->points) {
                float val1 = (axis1 == 0) ? point.x : (axis1 == 1) ? point.y : point.z;
                float val2 = (axis2 == 0) ? point.x : (axis2 == 1) ? point.y : point.z;
                
                if (std::isfinite(val1) && std::isfinite(val2)) {
                    min_val1 = std::min(min_val1, val1);
                    max_val1 = std::max(max_val1, val1);
                    min_val2 = std::min(min_val2, val2);
                    max_val2 = std::max(max_val2, val2);
                }
            }
            
            // Calculate scaling factors
            float scale_x = image_width * 0.9f / (max_val1 - min_val1);
            float scale_y = image_height * 0.9f / (max_val2 - min_val2);
            float scale = std::min(scale_x, scale_y);
            
            // Calculate offsets to center the projection
            float offset_x = (image_width - (max_val1 - min_val1) * scale) / 2;
            float offset_y = (image_height - (max_val2 - min_val2) * scale) / 2;
            
            // Draw points with error colors
            for (size_t i = 0; i < cloud->size(); i++) {
                const auto& point = cloud->points[i];
                if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                    continue;
                    
                // Extract coordinate values based on current projection
                float val1 = (axis1 == 0) ? point.x : (axis1 == 1) ? point.y : point.z;
                float val2 = (axis2 == 0) ? point.x : (axis2 == 1) ? point.y : point.z;
                
                // Apply flipping if needed (for opposite views)
                if (flip_axis1) val1 = max_val1 - (val1 - min_val1);
                if (flip_axis2) val2 = max_val2 - (val2 - min_val2);
                
                // Map to image coordinates
                int x = static_cast<int>((val1 - min_val1) * scale + offset_x);
                int y = static_cast<int>((val2 - min_val2) * scale + offset_y);
                
                if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
                    // Map error to color using JET colormap
                    float normalized_error = std::min(errors[i] / max_error, 1.0f);
                    cv::Vec3b color;
                    
                    // Same coloring logic as before
                    if (normalized_error < 0.25) {
                        color[0] = 255;                                    // B
                        color[1] = static_cast<uchar>(normalized_error * 4 * 255);  // G
                        color[2] = 0;                                      // R
                    } else if (normalized_error < 0.5) {
                        color[0] = static_cast<uchar>((0.5 - normalized_error) * 4 * 255);  // B
                        color[1] = 255;                                    // G
                        color[2] = static_cast<uchar>((normalized_error - 0.25) * 4 * 255); // R
                    } else if (normalized_error < 0.75) {
                        color[0] = 0;                                      // B
                        color[1] = static_cast<uchar>((0.75 - normalized_error) * 4 * 255); // G
                        color[2] = 255;                                    // R
                    } else {
                        color[0] = 0;                                      // B
                        color[1] = 0;                                      // G
                        color[2] = 255;                                    // R
                    }
                    
                    // Draw a circle at the point's position
                    cv::circle(error_image, cv::Point(x, y), 1, cv::Scalar(color[0], color[1], color[2]), -1);
                }
            }
            
            // Add view label
            cv::putText(error_image, view_names[view], 
                       cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 
                       1.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
            
            views[view] = error_image;
        }
        
        return views;
    }

public:
    // Main evaluation function
    EvaluationMetrics evaluateReconstruction(
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& ground_truth,
        const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& reconstruction,
        const std::string& output_dir = "",
        const std::string& prefix = "evaluation") {
            
        EvaluationMetrics metrics;
        metrics.valid_points = 0;
        
        // 1. Filter invalid points
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_gt(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_recon(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        std::vector<int> mapping;
        pcl::removeNaNFromPointCloud(*ground_truth, *filtered_gt, mapping);
        pcl::removeNaNFromPointCloud(*reconstruction, *filtered_recon, mapping);
        
        if (filtered_gt->empty() || filtered_recon->empty()) {
            std::cerr << "Error: Empty point clouds after filtering!" << std::endl;
            return metrics;
        }
        
        // 2. Build KD-trees for nearest neighbor search
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_gt;
        kdtree_gt.setInputCloud(filtered_gt);
        
        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree_recon;
        kdtree_recon.setInputCloud(filtered_recon);
        
        // 3. Calculate metrics
        double sum_distances_r2g = 0.0;
        double sum_distances_g2r = 0.0;
        double max_distance_r2g = 0.0;
        double max_distance_g2r = 0.0;
        std::vector<float> point_errors_recon;
        std::vector<float> point_errors_gt;
        
        // Histogram bins for error distribution (0 to 50mm in 1mm increments)
        metrics.error_histogram.resize(50, 0);
        
        // For each point in reconstruction, find distance to nearest point in ground truth
        int re_matches_1mm = 0;   // New counter for 1mm
        int re_matches_2_5mm = 0; // New counter for 2.5mm
        int re_matches_5mm = 0;
        int re_matches_10mm = 0;
        
        for (const auto& point : filtered_recon->points) {
            std::vector<int> indices(1);
            std::vector<float> sqr_distances(1);
            
            if (kdtree_gt.nearestKSearch(point, 1, indices, sqr_distances) > 0) {
                float distance = std::sqrt(sqr_distances[0]);
                
                sum_distances_r2g += sqr_distances[0];
                max_distance_r2g = std::max(max_distance_r2g, (double)distance);
                point_errors_recon.push_back(distance);
                
                // Update histogram
                int bin = std::min(static_cast<int>(distance * 1000), 49);
                metrics.error_histogram[bin]++;
                
                // Count matches within thresholds
                if (distance <= 0.001) re_matches_1mm++;     // New 1mm threshold
                if (distance <= 0.0025) re_matches_2_5mm++;  // New 2.5mm threshold
                if (distance <= 0.005) re_matches_5mm++;
                if (distance <= 0.010) re_matches_10mm++;
                
                metrics.valid_points++;
            }
        }

        // Calculate precision (how many reconstructed points are close to ground truth)
        metrics.precision_1mm = filtered_recon->size() > 0 ? 
            static_cast<double>(re_matches_1mm) / filtered_recon->size() : 0.0;

        metrics.precision_2_5mm = filtered_recon->size() > 0 ? 
            static_cast<double>(re_matches_2_5mm) / filtered_recon->size() : 0.0;

        metrics.precision_5mm = filtered_recon->size() > 0 ? 
            static_cast<double>(re_matches_5mm) / filtered_recon->size() : 0.0;

        metrics.precision_10mm = filtered_recon->size() > 0 ? 
            static_cast<double>(re_matches_10mm) / filtered_recon->size() : 0.0;
        
        // For completeness, check how many ground truth points have a nearby reconstruction point
        int gt_matched_points = 0;
        int gt_matches_1mm = 0;    // New counter for 1mm
        int gt_matches_2_5mm = 0;  // New counter for 2.5mm
        int gt_matches_5mm = 0;
        int gt_matches_10mm = 0;
        int valid_gt_points = 0;

        for (const auto& point : filtered_gt->points) {
            std::vector<int> indices(1);
            std::vector<float> sqr_distances(1);
            
            if (kdtree_recon.nearestKSearch(point, 1, indices, sqr_distances) > 0) {
                float distance = std::sqrt(sqr_distances[0]);

                // Only count as matched if within reasonable distance
                sum_distances_g2r += sqr_distances[0];
                max_distance_g2r = std::max(max_distance_g2r, (double)distance);
                point_errors_gt.push_back(distance);
                
                if (distance <= 0.001) gt_matches_1mm++;     // New 1mm threshold
                if (distance <= 0.0025) gt_matches_2_5mm++;  // New 2.5mm threshold
                if (distance <= 0.005) gt_matches_5mm++;
                if (distance <= 0.010) gt_matches_10mm++;

                valid_gt_points++;
            }
        }
        
        // Calculate recall (how many ground truth points are covered)
        metrics.recall_1mm = filtered_gt->size() > 0 ? 
            static_cast<double>(gt_matches_1mm) / filtered_gt->size() : 0.0;

        metrics.recall_2_5mm = filtered_gt->size() > 0 ? 
            static_cast<double>(gt_matches_2_5mm) / filtered_gt->size() : 0.0;

        metrics.recall_5mm = filtered_gt->size() > 0 ? 
            static_cast<double>(gt_matches_5mm) / filtered_gt->size() : 0.0;

        metrics.recall_10mm = filtered_gt->size() > 0 ? 
            static_cast<double>(gt_matches_10mm) / filtered_gt->size() : 0.0;
        
        // 4. Calculate final metrics
        metrics.chamfer_distance = metrics.valid_points > 0 ? (sum_distances_r2g + sum_distances_g2r)/(metrics.valid_points+valid_gt_points): -1;
        metrics.hausdorff_distance = std::max(max_distance_r2g, max_distance_g2r);
        metrics.rmse_r2g = metrics.valid_points > 0 ? std::sqrt(sum_distances_r2g / metrics.valid_points) : -1;
        metrics.rmse_g2r = metrics.valid_points > 0 ? std::sqrt(sum_distances_g2r / metrics.valid_points) : -1;
        
        // F-score at 5mm and 10mm thresholds
        metrics.f_score_1mm = (metrics.precision_1mm + metrics.recall_1mm > 0) ? 
            2 * metrics.precision_1mm * metrics.recall_1mm / (metrics.precision_1mm + metrics.recall_1mm) : 0;
            
        metrics.f_score_2_5mm = (metrics.precision_2_5mm + metrics.recall_2_5mm > 0) ? 
            2 * metrics.precision_2_5mm * metrics.recall_2_5mm / (metrics.precision_2_5mm + metrics.recall_2_5mm) : 0;
            
        metrics.f_score_5mm = (metrics.precision_5mm + metrics.recall_5mm > 0) ? 
            2 * metrics.precision_5mm * metrics.recall_5mm / (metrics.precision_5mm + metrics.recall_5mm) : 0;
            
        metrics.f_score_10mm = (metrics.precision_10mm + metrics.recall_10mm > 0) ? 
            2 * metrics.precision_10mm * metrics.recall_10mm / (metrics.precision_10mm + metrics.recall_10mm) : 0;
        
        // 5. Generate visualization if output directory is specified
        if (!output_dir.empty()) {
            if (!boost::filesystem::exists(output_dir)) {
                boost::filesystem::create_directories(output_dir);
            }
            
            // Create error visualization
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*filtered_recon));
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr error_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*filtered_gt));
            std::vector<cv::Mat> error_views_recon = createSixSidedErrorView(filtered_recon, point_errors_recon, 0.01);
            std::vector<cv::Mat> error_views_ground = createSixSidedErrorView(filtered_gt, point_errors_gt, 0.01);

            // Save individual views
            for (int i = 0; i < error_views_recon.size(); i++) {
                std::string view_name = (i == 0) ? "top" : 
                                       (i == 1) ? "front" : 
                                       (i == 2) ? "right" : 
                                       (i == 3) ? "bottom" : 
                                       (i == 4) ? "back" : "left";
                
                cv::Mat with_scale = addColorScale(error_views_recon[i], 0.0, 0.01);
                cv::imwrite(output_dir + "/" + prefix + "_error_reconstruct" + view_name + ".png", with_scale);
            }

            // Optionally create a combined view (2x3 grid of all views)
            int grid_width = error_views_recon[0].cols;
            int grid_height = error_views_recon[0].rows;
            cv::Mat combined(grid_height * 2, grid_width * 3, CV_8UC3, cv::Scalar(0, 0, 0));

            for (int i = 0; i < 6; i++) {
                int row = i / 3;
                int col = i % 3;
                error_views_recon[i].copyTo(combined(cv::Rect(col * grid_width, row * grid_height, grid_width, grid_height)));
            }

            cv::imwrite(output_dir + "/" + prefix + "_error_all_views.png", combined);

            for (int i = 0; i < error_views_ground.size(); i++) {
                std::string view_name = (i == 0) ? "top" : 
                                       (i == 1) ? "front" : 
                                       (i == 2) ? "right" : 
                                       (i == 3) ? "bottom" : 
                                       (i == 4) ? "back" : "left";
                
                cv::Mat with_scale = addColorScale(error_views_ground[i], 0.0, 0.01);
                cv::imwrite(output_dir + "/" + prefix + "_error_ground_" + view_name + ".png", with_scale);
            }

            // Optionally create a combined view (2x3 grid of all views)
            grid_height = error_views_ground[0].rows;
            grid_width = error_views_ground[0].cols;
            cv::Mat combined_ground(grid_height * 2, grid_width * 3, CV_8UC3, cv::Scalar(0, 0, 0));

            for (int i = 0; i < 6; i++) {
                int row = i / 3;
                int col = i % 3;
                error_views_ground[i].copyTo(combined_ground(cv::Rect(col * grid_width, row * grid_height, grid_width, grid_height)));
            }

            cv::imwrite(output_dir + "/" + prefix + "_error_ground_all_views.png", combined_ground);

            // Create detailed metrics report
            cv::Mat report = cv::Mat(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));
            int y_pos = 50;
            int line_height = 30;
            
            // Title
            cv::putText(report, "Point Cloud Reconstruction Quality Report", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_DUPLEX, 
                1.0, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            y_pos += line_height * 2;
            
            // Dataset info
            cv::putText(report, "Ground Truth: " + std::to_string(filtered_gt->size()) + " points", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            y_pos += line_height;
            
            cv::putText(report, "Reconstruction: " + std::to_string(filtered_recon->size()) + " points", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            y_pos += line_height * 1.5;
            
            // Accuracy metrics
            cv::putText(report, "Distance Metrics:", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            y_pos += line_height;
            
            // Replace bullet points with dashes
            cv::putText(report, "  - Chamfer Error: " + 
                std::to_string(metrics.chamfer_distance * 1000) + " mm", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 100, 0), 1, cv::LINE_AA);
            y_pos += line_height;
            
            cv::putText(report, "  - Hausdorff Error: " + 
                std::to_string(metrics.hausdorff_distance * 1000) + " mm", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 100, 0), 1, cv::LINE_AA);
            y_pos += line_height;
            
            cv::putText(report, "  - RMSE: " + 
                std::to_string(metrics.rmse_r2g * 1000) + " mm (Recon to GT)",
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 100, 0), 1, cv::LINE_AA);
            y_pos += line_height;

            cv::putText(report, "  - RMSE: " + 
                std::to_string(metrics.rmse_g2r * 1000) + " mm (GT to Recon)",
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 100, 0), 1, cv::LINE_AA);
            y_pos += line_height;
            
            // Coverage metrics
            cv::putText(report, "Coverage Metrics:", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            y_pos += line_height;
            
            cv::putText(report, "  - F-score (1mm): " + std::to_string(metrics.f_score_1mm), 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;

            cv::putText(report, "  - F-score (2.5mm): " + std::to_string(metrics.f_score_2_5mm), 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;

            cv::putText(report, "  - F-score (5mm): " + std::to_string(metrics.f_score_5mm), 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;
            
            cv::putText(report, "  - F-score (10mm): " + std::to_string(metrics.f_score_10mm), 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;
            
            cv::putText(report, "  - Recall (1mm): " + std::to_string(metrics.recall_1mm * 100) + "%", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;

            cv::putText(report, "  - Recall (2.5mm): " + std::to_string(metrics.recall_2_5mm * 100) + "%", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;

            cv::putText(report, "  - Recall (5mm) (groundtruth is covered): " + std::to_string(metrics.recall_5mm * 100) + "%", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;

            cv::putText(report, "  - Recall (10mm): " + std::to_string(metrics.recall_10mm * 100) + "%", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;
            
            cv::putText(report, "  - Precision (1mm): " + std::to_string(metrics.precision_1mm * 100) + "%",
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;

            cv::putText(report, "  - Precision (2.5mm): " + std::to_string(metrics.precision_2_5mm * 100) + "%",
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;

            cv::putText(report, "  - Precision (5mm reconstruct is right): " + std::to_string(metrics.precision_5mm * 100) + "%",
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;
            
            cv::putText(report, "  - Precision (10mm): " + std::to_string(metrics.precision_10mm * 100) + "%",
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(0, 0, 150), 1, cv::LINE_AA);
            y_pos += line_height;

            // Error distribution
            cv::putText(report, "Error Distribution:", 
                cv::Point(50, y_pos), cv::FONT_HERSHEY_SIMPLEX, 
                0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            y_pos += line_height * 1.5;
            
            // Draw histogram
            int hist_width = 600;
            int hist_height = 150;
            int hist_x = 100;
            int hist_y = y_pos;
            
            // Find max bin value for scaling - add safety check
            float max_bin = 1.0f;  // Default to 1.0 to avoid division by zero
            for (const auto& bin_value : metrics.error_histogram) {
                if (bin_value > max_bin) {
                    max_bin = bin_value;
                }
            }
            
            // Draw axes
            cv::line(report, cv::Point(hist_x, hist_y + hist_height), 
                     cv::Point(hist_x + hist_width, hist_y + hist_height), 
                     cv::Scalar(0, 0, 0), 1);
            cv::line(report, cv::Point(hist_x, hist_y), 
                     cv::Point(hist_x, hist_y + hist_height), 
                     cv::Scalar(0, 0, 0), 1);
            
            // Draw bars (plot first 30mm of errors)
            int num_bins_to_plot = std::min(30, static_cast<int>(metrics.error_histogram.size()));
            float bin_width = static_cast<float>(hist_width) / num_bins_to_plot;
            
            // Draw histogram bars with a minimum height to ensure visibility
            for (int i = 0; i < num_bins_to_plot; i++) {
                // Ensure at least 1 pixel height for non-zero bins
                float bin_height = (metrics.error_histogram[i] > 0) ? 
                    std::max(1.0f, metrics.error_histogram[i] / max_bin * hist_height) : 0;
                
                cv::rectangle(report, 
                             cv::Point(hist_x + i * bin_width, hist_y + hist_height - bin_height),
                             cv::Point(hist_x + (i + 1) * bin_width - 1, hist_y + hist_height),
                             cv::Scalar(255, 0, 0), -1);
                
                // Add border to each bar for better visibility
                cv::rectangle(report, 
                             cv::Point(hist_x + i * bin_width, hist_y + hist_height - bin_height),
                             cv::Point(hist_x + (i + 1) * bin_width - 1, hist_y + hist_height),
                             cv::Scalar(200, 0, 0), 1);
            }
            
            // X-axis labels
            for (int i = 0; i <= num_bins_to_plot; i += 5) {
                cv::line(report, 
                        cv::Point(hist_x + i * bin_width, hist_y + hist_height),
                        cv::Point(hist_x + i * bin_width, hist_y + hist_height + 5),
                        cv::Scalar(0, 0, 0), 1);
                        
                cv::putText(report, std::to_string(i), 
                           cv::Point(hist_x + i * bin_width - 5, hist_y + hist_height + 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
            
            cv::putText(report, "Error (mm)", 
                       cv::Point(hist_x + hist_width/2 - 30, hist_y + hist_height + 40),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            
            cv::imwrite(output_dir + "/" + prefix + "_report.png", report);
            
            // Save the raw metrics to a CSV file
            std::ofstream csv_file(output_dir + "/" + prefix + "_metrics.csv");
            csv_file << "Metric,Value\n";
            csv_file << "chamfer_distance," << metrics.chamfer_distance << "\n";
            csv_file << "hausdorff_distance," << metrics.hausdorff_distance << "\n";
            csv_file << "rmse_r2g," << metrics.rmse_r2g << "\n";
            csv_file << "rmse_g2r," << metrics.rmse_g2r << "\n";
            csv_file << "f_score_1mm," << metrics.f_score_1mm << "\n";
            csv_file << "f_score_2_5mm," << metrics.f_score_2_5mm << "\n";
            csv_file << "f_score_5mm," << metrics.f_score_5mm << "\n";
            csv_file << "f_score_10mm," << metrics.f_score_10mm << "\n";
            csv_file << "recall_1mm," << metrics.recall_1mm << "\n";
            csv_file << "recall_2_5mm," << metrics.recall_2_5mm << "\n";
            csv_file << "recall_5mm," << metrics.recall_5mm << "\n";
            csv_file << "recall_10mm," << metrics.recall_10mm << "\n";
            csv_file << "precision_1mm," << metrics.precision_1mm << "\n";
            csv_file << "precision_2_5mm," << metrics.precision_2_5mm << "\n";
            csv_file << "precision_5mm," << metrics.precision_5mm << "\n";
            csv_file << "precision_10mm," << metrics.precision_10mm << "\n";
            csv_file << "valid_points," << metrics.valid_points << "\n";
            csv_file.close();
        }
        
        return metrics;
    }
};

// Main function
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " ground_truth.pcd reconstruction.pcd [output_dir] [prefix]" << std::endl;
        return 1;
    }
    
    std::string gt_file = argv[1];
    std::string recon_file = argv[2];
    std::string output_dir = (argc > 3) ? argv[3] : "home/ok/mr-30/data/";
    std::string prefix = (argc > 4) ? argv[4] : "evaluation";
    
    // Load point clouds
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_truth(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr reconstruction(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(gt_file, *ground_truth) == -1) {
        std::cerr << "Could not load ground truth point cloud: " << gt_file << std::endl;
        return 1;
    }
    
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(recon_file, *reconstruction) == -1) {
        std::cerr << "Could not load reconstruction point cloud: " << recon_file << std::endl;
        return 1;
    }
    
    std::cout << "Loaded point clouds:\n";
    std::cout << "  Ground truth: " << ground_truth->size() << " points\n";
    std::cout << "  Reconstruction: " << reconstruction->size() << " points\n";
    
    // Evaluate reconstruction
    PointCloudEvaluator evaluator;
    EvaluationMetrics metrics = evaluator.evaluateReconstruction(ground_truth, reconstruction, output_dir, prefix);
    
    // Print results
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "\n=== Reconstruction Quality Metrics ===\n";
    std::cout << "\n--- Distance Metrics (LOWER is better, perfect = 0mm) ---\n";
    std::cout << "Chamfer Distance: " << metrics.chamfer_distance * 1000 << " mm\n";
    std::cout << "Hausdorff Distance: " << metrics.hausdorff_distance * 1000 << " mm\n";
    std::cout << "RMSE reconstruction to ground truth: " << metrics.rmse_r2g * 1000 << " mm\n";
    std::cout << "RMSE ground truth to reconstruction: " << metrics.rmse_g2r * 1000 << " mm\n";

    std::cout << "\n--- Coverage Metrics (HIGHER is better, perfect = 1.0) ---\n";
    std::cout << "F-Score (1mm): " << metrics.f_score_1mm << "\n";
    std::cout << "F-Score (2.5mm): " << metrics.f_score_2_5mm << "\n";
    std::cout << "F-Score (5mm): " << metrics.f_score_5mm << "\n";
    std::cout << "F-Score (10mm): " << metrics.f_score_10mm << "\n";

    std::cout << "\n--- Accuracy Metrics (HIGHER is better, perfect = 100%) ---\n";
    std::cout << "Precision (1mm): " << metrics.precision_1mm * 100 << "%\n";
    std::cout << "Precision (2.5mm): " << metrics.precision_2_5mm * 100 << "%\n";
    std::cout << "Precision (5mm): " << metrics.precision_5mm * 100 << "%\n";
    std::cout << "Precision (10mm): " << metrics.precision_10mm * 100 << "%\n";
    std::cout << "Recall (1mm): " << metrics.recall_1mm * 100 << "%\n";
    std::cout << "Recall (2.5mm): " << metrics.recall_2_5mm * 100 << "%\n";
    std::cout << "Recall (5mm): " << metrics.recall_5mm * 100 << "%\n";
    std::cout << "Recall (10mm): " << metrics.recall_10mm * 100 << "%\n";

    std::cout << "\nValid Points: " << metrics.valid_points << "\n";

    if (!output_dir.empty()) {
        std::cout << "\nVisualization and detailed report saved to " << output_dir << "\n";
    }

    return 0;
}

// Precision is how accurate the reconstructed points are
// Recall is how many of the ground truth points are covered by the reconstruction