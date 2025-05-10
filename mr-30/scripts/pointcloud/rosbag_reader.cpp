#include "rosbag_reader.hpp"

RosbagReader::RosbagReader() 
    : tf_buffer(ros::Duration(10000)),
    tf_listener(tf_buffer),
    generator()
{}

std::vector<FrameData> RosbagReader::readBag(const std::string& bag_file, 
                                             const std::string& output_path_image,
                                             const cameraNoiseConfig& camera_config, 
                                             const robotNoiseConfig& robot_config,
                                             const int number_of_frames,
                                             const std::string& point_cloud_topic) {
    rosbag::Bag bag;
    // Open the bag file
    try {
        bag.open(bag_file, rosbag::bagmode::Read);
    } 
    catch (const rosbag::BagException& e) {
        ROS_ERROR_STREAM("Failed to read bag: " << e.what());
        return std::vector<FrameData>();
    }

    // check if output path exists, else create it
    if (!boost::filesystem::exists(output_path_image)) {
        boost::filesystem::create_directories(output_path_image);
    }

    // Define required topics
    std::vector<std::string> required_topics = {
        point_cloud_topic,
        "/tf",
        "/tf_static"
    };

    // Create a view to iterate over the messages in the bag
    rosbag::View view(bag, rosbag::TopicQuery(required_topics));

    // Process TF messages to populate the TF buffer
    for (const rosbag::MessageInstance& m : view) {
        if (m.getTopic() == "/tf" || m.getTopic() == "/tf_static") {
            auto tf_msg = m.instantiate<tf2_msgs::TFMessage>();
            if (tf_msg) {
                for (const auto& transform : tf_msg->transforms) {
                    tf_buffer.setTransform(transform, "default_authority", m.getTopic() == "/tf_static");
                }
            }
        }
    }

    // Process point cloud messages and corresponding transforms
    std::vector<FrameData> frame_data;
    int frame_idx = 0;
    for (const rosbag::MessageInstance& m : view) {
        if (m.getTopic() == point_cloud_topic) {
            auto point_cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (point_cloud_msg) {
                geometry_msgs::TransformStamped transform_stamped;
                try {
                    // Lookup the transform from the camera frame to the global frame
                    transform_stamped = tf_buffer.lookupTransform("world", point_cloud_msg->header.frame_id, point_cloud_msg->header.stamp);
                    
                } catch (const tf2::TransformException& ex) {
                    ROS_WARN_STREAM("Transform failed: " << ex.what());
                    
                    // set the transform to zero if it fails
                    transform_stamped.transform.translation.x = 0;
                    transform_stamped.transform.translation.y = 0;
                    transform_stamped.transform.translation.z = 0;
                    transform_stamped.transform.rotation.x = 0;
                    transform_stamped.transform.rotation.y = 0;
                    transform_stamped.transform.rotation.z = 0;
                    transform_stamped.transform.rotation.w = 1;
                } 

                pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::fromROSMsg(*point_cloud_msg, *pcl_cloud);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZRGB>(*pcl_cloud));

                std::string frame_prefix = "frame" + std::to_string(frame_idx) + "_";
                saveDepthAsImage(pcl_cloud, output_path_image, frame_prefix + "depth_image");

                // Process the point cloud and transform here
                if (camera_config.enabled) {
                    cameraNoise(pcl_cloud, camera_config);
                }
                if (camera_config.use_perlin) { 
                    cameraPerlinNoise(pcl_cloud, camera_config);
                }
                if (robot_config.enabled && frame_idx > 0) { // skip first frame, so ground truth is in same global frame for more accurate comparison
                    robotNoise(transform_stamped.transform, robot_config);
                }

                saveDepthAsImage(pcl_cloud, output_path_image, frame_prefix +  "depth_image_noisy");

                saveAmplifiedDifferenceImage(original_cloud, pcl_cloud, output_path_image, frame_prefix + "depth_image_diff");

                FrameData data;
                data.point_cloud = pcl_cloud;
                data.transform = transform_stamped.transform;
                data.timestamp = point_cloud_msg->header.stamp;
                frame_data.push_back(data);
                frame_idx++;
                
                if (frame_idx >= number_of_frames) {
                    ROS_INFO("Reached maximum frames (%d), stopping processing", number_of_frames);
                    break;
                }
            }
        }
    }
    calculateTrajectoryMetrics(frame_data);
    bag.close();
    return frame_data;
}

void RosbagReader::robotNoise(geometry_msgs::Transform& transform, const robotNoiseConfig& config) {
    float translation_stddev = config.tran_stddev; // Standard deviation for translation noise
    float rotation_stddev = config.rot_stddev;    // Standard deviation for rotation noise

    std::normal_distribution<float> translation_noise(config.mean, translation_stddev);
    std::normal_distribution<float> rotation_noise(config.mean, rotation_stddev);

    // Add noise to translation
    transform.translation.x += translation_noise(generator);
    transform.translation.y += translation_noise(generator);
    transform.translation.z += translation_noise(generator);

    // Convert quaternion to tf2::Quaternion for easier manipulation
    tf2::Quaternion quat;
    tf2::fromMsg(transform.rotation, quat);

    // Add noise to rotation
    double roll, pitch, yaw;
    tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
    roll += rotation_noise(generator);
    pitch += rotation_noise(generator);
    yaw += rotation_noise(generator);

    // Convert back to quaternion
    quat.setRPY(roll, pitch, yaw);
    transform.rotation = tf2::toMsg(quat);
}

// change so it takes in pcl::PointCloud<pcl::PointXYZRGB>::Ptr instead of ros msg 
void RosbagReader::cameraNoise(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const cameraNoiseConfig& config) {
    float base_stddev_ = config.base_stddev;
    float scale_factor_ = config.scale_factor;
    ROS_INFO("Camera noise enabled with base stddev: %.3f, scale factor: %.3f", base_stddev_, scale_factor_);

    // Apply Gaussian noise to the depth (z) field only
    for (auto& point : cloud->points) {
        if (!std::isfinite(point.z))
            continue; // Skip invalid depth values

        // Scale the noise based on the depth value
        float scaled_stddev = base_stddev_ + scale_factor_ * point.z;
        std::normal_distribution<float> depth_noise(config.mean, scaled_stddev);

        // Add noise to the z coordinate (depth)
        point.z += depth_noise(generator);
    }
}

void RosbagReader::cameraPerlinNoise(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, const cameraNoiseConfig& config) {
    // Perlin noise implementation based on Improved Perlin Noise by Ken Perlin
    
    // Permutation table for Perlin noise
    static const int permutation[] = {
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
        140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
        247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
        57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
        74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
        60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
        65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
        200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
        52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
        207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
        119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
        129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
        218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
        81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
        184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
        222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
    };
    
    // Create extended permutation table
    static int p[512];
    static bool initialized = false;
    if (!initialized) {
        for(int i = 0; i < 256; i++) {
            p[256 + i] = p[i] = permutation[i];
        }
        initialized = true;
    }
    
    // Helper functions for Perlin noise
    auto fade = [](float t) -> float {
        // Improved fade function: 6t^5 - 15t^4 + 10t^3
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    };
    
    auto lerp = [](float t, float a, float b) -> float {
        return a + t * (b - a);
    };
    
    auto grad = [&p](int hash, float x, float y, float z) -> float {
        int h = hash & 15;                      // CONVERT LO 4 BITS OF HASH CODE
        float u = h < 8 ? x : y,                // INTO 12 GRADIENT DIRECTIONS
              v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    };
    
    // Perlin noise function
    auto perlin = [&p, &fade, &lerp, &grad](float x, float y, float z) -> float {
        // Find unit cube that contains the point
        int X = static_cast<int>(std::floor(x)) & 255;
        int Y = static_cast<int>(std::floor(y)) & 255;
        int Z = static_cast<int>(std::floor(z)) & 255;
        
        // Find relative x, y, z of point in cube
        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);
        
        // Compute fade curves for each coordinate
        float u = fade(x);
        float v = fade(y);
        float w = fade(z);
        
        // Hash coordinates of the 8 cube corners
        int A = p[X] + Y;
        int AA = p[A] + Z;
        int AB = p[A + 1] + Z;
        int B = p[X + 1] + Y;
        int BA = p[B] + Z;
        int BB = p[B + 1] + Z;
        
        // Add blended results from 8 corners of cube
        return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                    grad(p[BA], x - 1, y, z)),
                           lerp(u, grad(p[AB], x, y - 1, z),
                                grad(p[BB], x - 1, y - 1, z))),
                   lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1),
                                grad(p[BA + 1], x - 1, y, z - 1)),
                        lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
                             grad(p[BB + 1], x - 1, y - 1, z - 1))));
    };
    
    // Perlin noise application parameters
    float scale = config.perline_scale;         // Controls size of noise features
    float amplitude = config.perlin_amplitude;     // Controls magnitude of noise
    int octaves = config.perlin_nr_layers;                          // Number of noise layers to combine
    
    // Generate a random offset for this cloud to make each frame's noise unique
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    float offsetX = dist(generator);
    float offsetY = dist(generator);
    float offsetZ = dist(generator);
    
    // Apply noise to each point
    for (auto& point : cloud->points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue; // Skip invalid points
        
        float noise = 0.0f;
        float max_amp = 0.0f;
        float freq = 1.0f;
        float amp = 1.0f;
        float scale_factor = 0.01f;
        
        // Sum multiple octaves of noise
        for (int i = 0; i < octaves; ++i) {
            // Calculate noise based on position
            float sample_x = (point.x * scale * freq) + offsetX;
            float sample_y = (point.y * scale * freq) + offsetY;
            float sample_z = (point.z * scale * freq) + offsetZ;
            
            noise += perlin(sample_x, sample_y, sample_z) * amp;
            
            max_amp += amp;
            freq *= 2.0f;   // Double frequency each octave
            amp *= 0.5f;    // Half amplitude each octave
        }
        
        // Normalize noise to [-1, 1] range
        noise /= max_amp;
        
        // Scale noise by depth - more noise for farther objects
        float depth_scale = 1.0f + (point.z * scale_factor);
        
        // Apply noise to z coordinate
        point.z += noise * amplitude * depth_scale;
    }
}

void RosbagReader::saveDepthAsImage(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, 
                                    const std::string& output_path_image,
                                   const std::string& filename) {
    if (cloud->empty()) {
        ROS_WARN("Cannot save empty point cloud as image");
        return;
    }
    
    // Check if cloud is organized (has image structure)
    bool is_organized = cloud->height > 1;
    cv::Mat depth_image;
    
    if (is_organized) {
        // For organized clouds, directly map to image
        depth_image = cv::Mat(cloud->height, cloud->width, CV_32F, cv::Scalar(0));
        
        // Extract depth values
        for (int h = 0; h < cloud->height; h++) {
            for (int w = 0; w < cloud->width; w++) {
                const auto& point = cloud->at(w, h);
                if (std::isfinite(point.z)) {
                    depth_image.at<float>(h, w) = point.z;
                }
            }
        }
    } else {
        // For unorganized clouds, find bounds and create a projection
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        
        int width = 640;  // Default image width
        int height = 480; // Default image height
        
        depth_image = cv::Mat(height, width, CV_32F, cv::Scalar(0));
        
        // Simple projection of points to image plane
        for (const auto& point : cloud->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                continue;
            
            // Map x,y to image coordinates
            int x = static_cast<int>((point.x - min_pt.x) / (max_pt.x - min_pt.x) * (width - 1));
            int y = static_cast<int>((point.y - min_pt.y) / (max_pt.y - min_pt.y) * (height - 1));
            
            if (x >= 0 && x < width && y >= 0 && y < height) {
                // Use the smallest depth if multiple points project to same pixel
                float& current_depth = depth_image.at<float>(y, x);
                if (current_depth == 0 || point.z < current_depth) {
                    current_depth = point.z;
                }
            }
        }
    }
    
    // Find valid min/max depth values for normalization
    float min_depth = std::numeric_limits<float>::max();
    float max_depth = 0;
    for (int i = 0; i < depth_image.rows; i++) {
        for (int j = 0; j < depth_image.cols; j++) {
            float depth = depth_image.at<float>(i, j);
            if (depth > 0) {
                min_depth = std::min(min_depth, depth);
                max_depth = std::max(max_depth, depth);
            }
        }
    }
    
    // Normalize depths to 0-255 range
    cv::Mat depth_image_8u;
    if (max_depth > min_depth) {
        depth_image = (depth_image - min_depth) / (max_depth - min_depth);
        depth_image.convertTo(depth_image_8u, CV_8UC1, 255.0);
        //cv::equalizeHist(depth_image_8u, depth_image_8u);
    } else {
        depth_image.convertTo(depth_image_8u, CV_8UC1);
    }
    
    // Save only grayscale image
    cv::imwrite(output_path_image + filename + ".png", depth_image_8u);
}

void RosbagReader::saveAmplifiedDifferenceImage(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& original_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& noisy_cloud,
    const std::string& output_path_image,
    const std::string& filename) {
    
    // Create depth difference image directly (skip intermediate point cloud)
    cv::Mat diff_image(original_cloud->height, original_cloud->width, CV_32F, cv::Scalar(0));
    
    float amplification = 5.0f;
    float actual_max_diff = 0.0f;  // Track actual maximum difference for logging
    
    // Fixed scale: 8mm after removing amplification factor
    float fixed_max_meters = 0.006f;  // 8mm in meters
    float fixed_max_amplified = fixed_max_meters * amplification;  // Amplified value for internal use
    
    // Calculate differences directly into image
    int valid_points = 0;
    int over_limit_points = 0;
    
    for (int h = 0; h < original_cloud->height; h++) {
        for (int w = 0; w < original_cloud->width; w++) {
            const auto& orig_pt = original_cloud->at(w, h);
            const auto& noisy_pt = noisy_cloud->at(w, h);
            
            if (std::isfinite(orig_pt.z) && std::isfinite(noisy_pt.z)) {
                float diff = std::abs(noisy_pt.z - orig_pt.z) * amplification;
                diff_image.at<float>(h, w) = diff;
                
                // Track the true maximum difference
                actual_max_diff = std::max(actual_max_diff, diff);
                valid_points++;
                
                // Count points over our limit
                if (diff > fixed_max_amplified) {
                    over_limit_points++;
                }
            }
        }
    }

    // Use fixed scale for visualization
    float scale = std::max(fixed_max_amplified, 0.00001f);// Minimum scale to avoid division by zero
    
    // Convert to 8-bit and apply colormap
    cv::Mat depth_8u;
    diff_image.convertTo(depth_8u, CV_8UC1, 255.0/scale);
    
    cv::Mat depth_color;
    cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET);
    
    // Add colorbar using the fixed scale
    cv::Mat with_scale = addColorScale(depth_color, 0.0f, fixed_max_meters); // Use non-amplified value
    
    // Save the image
    cv::imwrite(output_path_image + filename + ".png", with_scale);
    
    // Log information about the differences
    float actual_max_meters = actual_max_diff / amplification;

    if (over_limit_points > 0) {
        ROS_WARN("%d points (%.1f%%) exceed the 8mm scale", 
                over_limit_points, 100.0f * over_limit_points / valid_points);
    }
}

cv::Mat RosbagReader::addColorScale(const cv::Mat& image, float min_val, float max_val) {
    // Create image with space for scale bar - increased sizes
    int bar_width = 40;       // Increased from 20
    int margin = 15;          // Increased from 10
    int border = bar_width + margin*3 + 80;  // Increased width for larger text + margins
    cv::Mat result(image.rows, image.cols + border, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Copy original image
    image.copyTo(result(cv::Rect(0, 0, image.cols, image.rows)));
    
    // Calculate colorbar position
    int bar_x = image.cols + margin;
    int bar_height = image.rows * 0.75;  // Increased from 0.7 (75% of image height)
    int bar_y = (image.rows - bar_height) / 2;
    
    // Draw colorbar
    for (int y = 0; y < bar_height; y++) {
        // Color from top (1.0) to bottom (0.0)
        float t = 1.0f - (float)y / bar_height;
        float value = min_val + t * (max_val - min_val);
        
        cv::rectangle(
            result, 
            cv::Point(bar_x, bar_y + y), 
            cv::Point(bar_x + bar_width, bar_y + y + 1),
            colorFromJet(t),
            -1  // filled
        );
        
        // Add labels (5 evenly spaced)
        if (y % (bar_height/4) == 0 || y == bar_height-1) {
            char label[10];
            snprintf(label, sizeof(label), "%.3f", value);
            cv::putText(
                result, label,
                cv::Point(bar_x + bar_width + 8, bar_y + y + 6),  // Adjusted position
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2  // Increased font size from 0.4 to 0.7, added thickness
            );
        }
    }
    
    // Add border and title
    cv::rectangle(result, 
                 cv::Point(bar_x, bar_y), 
                 cv::Point(bar_x + bar_width, bar_y + bar_height),
                 cv::Scalar(0, 0, 0), 2);  // Increased border thickness
    
    cv::putText(result, "Noise (m)", 
               cv::Point(bar_x, bar_y - 24),  // Adjusted position
               cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 0), 2);  // Increased font size from 0.5 to 0.9, added thickness
    
    return result;
}

// Helper to convert normalized value to JET colormap color
cv::Scalar RosbagReader::colorFromJet(float normalized) {
    // Simplified JET colormap implementation
    float r = std::min(std::max(1.5f - std::abs(normalized - 0.75f) * 4.0f, 0.0f), 1.0f);
    float g = std::min(std::max(1.5f - std::abs(normalized - 0.5f) * 4.0f, 0.0f), 1.0f);
    float b = std::min(std::max(1.5f - std::abs(normalized - 0.25f) * 4.0f, 0.0f), 1.0f);
    
    return cv::Scalar(b*255, g*255, r*255);
}

void RosbagReader::calculateTrajectoryMetrics(const std::vector<FrameData>& frame_data) {
    TrajectoryMetrics metrics = {0, 0.0, 0.0};
    
    if (frame_data.empty()) {
        ROS_WARN("Cannot calculate trajectory metrics: Empty frame data");
        return;
    }
    
    // Number of points (frames) in trajectory
    metrics.num_points = frame_data.size();
    
    // Prepare to write CSV file
    std::ofstream csv_file("/home/ok/mr30_ws/trajectory_data.csv");
    if (!csv_file.is_open()) {
        ROS_ERROR("Failed to open CSV file for writing");
    } else {
        // Write CSV header
        csv_file << "frame_id,timestamp,x,y,z,distance_from_start,distance_from_prev,cumulative_distance,elapsed_time\n";
    }
    
    // Calculate total distance traveled
    metrics.distance_traveled = 0.0;
    double start_time_seconds = 0.0;
    
    if (frame_data.size() >= 1) {
        const auto& start_time = frame_data.front().timestamp;
        start_time_seconds = start_time.sec + start_time.nsec / 1e9;
    }
    
    for (size_t i = 0; i < frame_data.size(); ++i) {
        const auto& curr_transform = frame_data[i].transform;
        float x = curr_transform.translation.x;
        float y = curr_transform.translation.y;
        float z = curr_transform.translation.z;
        
        // Calculate distances
        double dist_from_prev = 0.0;
        double dist_from_start = 0.0;
        
        if (i > 0) {
            const auto& prev_transform = frame_data[i-1].transform;
            
            // Distance from previous frame
            double dx = x - prev_transform.translation.x;
            double dy = y - prev_transform.translation.y;
            double dz = z - prev_transform.translation.z;
            
            dist_from_prev = std::sqrt(dx*dx + dy*dy + dz*dz);
            metrics.distance_traveled += dist_from_prev;
        }
        
        // Distance from start point
        if (i > 0) {
            const auto& start_transform = frame_data[0].transform;
            double dx = x - start_transform.translation.x;
            double dy = y - start_transform.translation.y;
            double dz = z - start_transform.translation.z;
            
            dist_from_start = std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        
        // Calculate timestamp difference from start
        const auto& timestamp = frame_data[i].timestamp;
        double time_seconds = timestamp.sec + timestamp.nsec / 1e9;
        double elapsed_time = time_seconds - start_time_seconds;
        
        // Write to CSV if file is open
        if (csv_file.is_open()) {
            csv_file << i << ","
                     << timestamp << ","
                     << x << ","
                     << y << ","
                     << z << ","
                     << dist_from_start << ","
                     << dist_from_prev << ","
                     << metrics.distance_traveled << ","
                     << elapsed_time << "\n";
        }
    }
    
    // Close CSV file
    if (csv_file.is_open()) {
        csv_file.close();
        ROS_INFO("Trajectory data saved to trajectory_data.csv");
    }
    
    // Calculate time elapsed
    if (frame_data.size() >= 2) {
        const auto& start_time = frame_data.front().timestamp;
        const auto& end_time = frame_data.back().timestamp;
        
        double start_seconds = start_time.sec + start_time.nsec / 1e9;
        double end_seconds = end_time.sec + end_time.nsec / 1e9;
        
        metrics.time_elapsed = end_seconds - start_seconds;
    }
    
    ROS_INFO("Trajectory metrics: %d points, %.3f meters, %.3f seconds", 
             metrics.num_points, metrics.distance_traveled, metrics.time_elapsed);
}