#include "utils/sophus_utils.hpp"
#include "IMU/imuPreintegrated.hpp"
#include "initMethod/drtVioInit.h"
#include "initMethod/drtLooselyCoupled.h"
#include "initMethod/drtTightlyCoupled.h"
#include "utils/eigenUtils.hpp"
#include "utils/ticToc.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <glog/logging.h>
#include <string>

using namespace std;
using namespace cv;

struct MotionData1
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double timestamp;
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;
    Eigen::Vector3d imu_acc;
    Eigen::Vector3d imu_gyro;

    Eigen::Vector3d imu_gyro_bias;
    Eigen::Vector3d imu_acc_bias;

    Eigen::Vector3d imu_velocity;
};

struct FeatureData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int feature_id;
    double x;
    double y;
    double z = 1;
    double p_u;
    double p_v;
    double velocity_x = 0;
    double velocity_y = 0;
};

struct GtData {
    double timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond rotation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d bias_gyr;
    Eigen::Vector3d bias_acc;
};

void LoadFeature(std::string filename, std::vector<FeatureData>& feature)
{

    std::ifstream f;
    f.open(filename.c_str());

    int feature_id=0;

    if(!f.is_open())
    {
        std::cerr << " can't open LoadFeatures file "<<std::endl;
        return;
    }

    while (!f.eof()) {

        std::string s;
        std::getline(f,s);


        if(! s.empty())
        {
            std::stringstream ss;
            ss << s;

            FeatureData data;
            Eigen::Vector4d point;
            Eigen::Vector4d obs;

            ss>>point[0];
            ss>>point[1];
            ss>>point[2];
            ss>>point[3];
            ss>>obs[0];
            ss>>obs[1];

            // 行号即id
            data.feature_id=feature_id;
            feature_id++;

            data.x=obs[0];
            data.y=obs[1];
            data.z=1;
            data.p_u=(obs[0]*460 + 255);
            data.p_v=(obs[1]*460 + 255);
            feature.push_back(data);

        }
    }

}

void LoadPose(std::string filename, std::vector<MotionData1>& pose, Eigen::map<double,GtData>& gt)
{

    std::ifstream f;
    f.open(filename.c_str());

    if(!f.is_open())
    {
        std::cerr << " can't open LoadPoses file "<<std::endl;
        return;
    }

    while (!f.eof()) {

        std::string s;
        std::getline(f,s);

        if(! s.empty())
        {
            std::stringstream ss;
            ss << s;

            MotionData1 data;
            GtData gtdata;
            double time;
            Eigen::Quaterniond q;
            Eigen::Vector3d t;
            Eigen::Vector3d gyro;
            Eigen::Vector3d acc;

            ss>>time;
            ss>>q.w();
            ss>>q.x();
            ss>>q.y();
            ss>>q.z();
            ss>>t(0);
            ss>>t(1);
            ss>>t(2);
            ss>>gyro(0);
            ss>>gyro(1);
            ss>>gyro(2);
            ss>>acc(0);
            ss>>acc(1);
            ss>>acc(2);


            data.timestamp = time;;
            data.imu_gyro = gyro;
            data.imu_acc = acc;
            data.twb = t;
            data.Rwb = Eigen::Matrix3d(q);
            pose.push_back(data);

            gtdata.timestamp=time;
            gtdata.position=t;
            gtdata.rotation=q;
            gtdata.velocity=Eigen::Vector3d::Zero();
            gtdata.bias_acc=Eigen::Vector3d::Zero();
            gtdata.bias_gyr=Eigen::Vector3d::Zero();

            gt.emplace(time,gtdata);

        }
    }

}

int main(int argc, char **argv) {


    bool use_single_ligt = false;
    bool use_ligt_vins = false;

    if (argc != 3) {
        std::cout << "Usage: code" << " code type" << " data type"  << "\n";
        return -1;
    }

    char *codeType = argv[1];
    char *dataType = argv[2];
    std::ofstream save_file("../result/" + string(codeType) + "_" + string(dataType) + ".txt");

    readParameters("../config/simulator1.yaml");
    PUB_THIS_FRAME = true;
    double sf = std::sqrt(double(IMU_FREQ));

    std::vector<MotionData1> cam_poses;
    Eigen::map<double,GtData> cam_gt;
    LoadPose("../simulator_data/cam_pose.txt",cam_poses,cam_gt);
    std::vector<MotionData1> imu_poses;
    Eigen::map<double,GtData> imu_gt;
    LoadPose("../simulator_data/imu_pose_noise.txt",imu_poses,imu_gt);



    for (int i = 0; i < cam_poses.size() - 100; i += 10) {

        DRT::drtVioInit::Ptr  pDrtVioInit;
        if (string(codeType) == "drtTightly")
        {
            pDrtVioInit.reset(new DRT::drtTightlyCoupled(RIC[0], TIC[0]));
        }

        if (string(codeType) == "drtLoosely")
        {
            pDrtVioInit.reset(new DRT::drtLooselyCoupled(RIC[0], TIC[0]));
        }


        std::vector<int> idx;

        // 40 4HZ, 0.25s
        for (int j = i; j < 100 + i; j += 1)
            idx.push_back(j);

        double last_img_t_s, cur_img_t_s;
        bool first_img = true;
        bool init_feature = true;


        std::vector<double> idx_time;

        for (int i: idx) {

            cur_img_t_s = cam_poses[i].timestamp;

            std::stringstream feature_filename;
            feature_filename<<"../simulator_data/keyframe/all_points_"<<i<<".txt";
            std::vector<FeatureData> features;
            LoadFeature(feature_filename.str(),features);

            Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 7, 1 >> >>
                    image;
            for (unsigned int i = 0; i < features.size(); i++) {
                int feature_id = features[i].feature_id;
                int camera_id = 0;
                double x = features[i].x;
                double y = features[i].y;
                double z = 1;
                double p_u = features[i].p_u;
                double p_v = features[i].p_v;
                double velocity_x = 0;
                double velocity_y = 0;
                assert(camera_id == 0);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
            }

            if (init_feature) {
                init_feature = false;
                continue;
            }

            if (pDrtVioInit->addFeatureCheckParallax(cur_img_t_s, image, 0.0)) {

                idx_time.push_back(cur_img_t_s);

                std::cout << "add image is: " << fixed << cur_img_t_s << " image number is: " << idx_time.size()
                          << std::endl;

                if (first_img) {
                    last_img_t_s = cur_img_t_s;
                    first_img = false;
                    continue;
                }

                std::vector<MotionData> imu_segment;

                for (size_t i = 0; i < imu_poses.size(); i++) {
                    double timestamp = imu_poses[i].timestamp;

                    MotionData imu_data;
                    imu_data.timestamp = timestamp;
                    imu_data.imu_acc = imu_poses[i].imu_acc;
                    imu_data.imu_gyro = imu_poses[i].imu_gyro;

                    if (timestamp > last_img_t_s && timestamp <= cur_img_t_s) {
                        imu_segment.push_back(imu_data);
                    }
                    if (timestamp > cur_img_t_s) {
                        imu_segment.push_back(imu_data);
                        break;
                    }
                }

                vio::IMUBias bias;
                vio::IMUCalibParam
                        imu_calib(RIC[0], TIC[0], GYR_N * sf, ACC_N * sf, GYR_W / sf, ACC_W / sf);
                vio::IMUPreintegrated imu_preint(bias, &imu_calib, last_img_t_s, cur_img_t_s);

                int n = imu_segment.size() - 1;

                for (int i = 0; i < n; i++) {
                    double dt;
                    Eigen::Vector3d gyro;
                    Eigen::Vector3d acc;

                    if (i == 0 && i < (n - 1))               // [start_time, imu[0].time]
                    {
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tini = imu_segment[i].timestamp - last_img_t_s;
                        CHECK(tini >= 0);
                        acc = (imu_segment[i + 1].imu_acc + imu_segment[i].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tini / tab)) * 0.5f;
                        gyro = (imu_segment[i + 1].imu_gyro + imu_segment[i].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tini / tab)) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - last_img_t_s;
                    } else if (i < (n - 1))      // [imu[i].time, imu[i+1].time]
                    {
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                    } else if (i > 0 && i == n - 1) {
                        // std::cout << " n : " << i + 1 << " " << n << " " << imu_segment[i + 1].timestamp << std::endl;
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tend = imu_segment[i + 1].timestamp - cur_img_t_s;
                        CHECK(tend >= 0);
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tend / tab)) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tend / tab)) * 0.5f;
                        dt = cur_img_t_s - imu_segment[i].timestamp;
                    } else if (i == 0 && i == (n - 1)) {
                        acc = imu_segment[i].imu_acc;
                        gyro = imu_segment[i].imu_gyro;
                        dt = cur_img_t_s - last_img_t_s;
                    }

                    CHECK(dt >= 0);
                    imu_preint.integrate_new_measurement(gyro, acc, dt);
                }
                // std::cout << fixed << "cur time: " << cur_img_t_s << " " << "last time: " << last_img_t_s << std::endl;

                pDrtVioInit->addImuMeasure(imu_preint);

                last_img_t_s = cur_img_t_s;


            }

            if (idx_time.size() >= 10) break;

        }

        if (idx_time.size() < 10) continue;

        bool is_good = pDrtVioInit->checkAccError();

        if (!is_good)
        {
            continue;
        }

        if ( !pDrtVioInit->process()) {
            save_file << "time: " << fixed << idx_time[0] << " other_reason" << std::endl;
            save_file << "scale_error: " << "nan" << std::endl;
            save_file << "pose_error: " << "nan" << std::endl;
            save_file << "biasg_error: " << "nan" << std::endl;
            save_file << "velo_error: " << "nan" << std::endl;
            save_file << "gravity_error: " << "nan" << " " << "nan" << std::endl;
            save_file << "v0_error: "  << "nan" << std::endl;
            save_file << "gt_vel_rot: " << "nan" << " " << "nan" << std::endl;
            LOG(INFO) << "---scale: ";
            std::cout << "time: " << fixed << idx_time[0] << std::endl;
            std::cout << "scale_error: " << 100 << std::endl;
            std::cout << "pose_error: " << 100 << std::endl;
            std::cout << "biasg_error: " << 100 << std::endl;
            std::cout << "velo_error: " << 100 << std::endl;
            std::cout << "rot_error: " << "nan" << std::endl;
            continue;
        }

        // 获取真实值
        std::vector<Eigen::Vector3d> gt_pos;
        std::vector<Eigen::Matrix3d> gt_rot;
        std::vector<Eigen::Vector3d> gt_vel;
        std::vector<Eigen::Vector3d> gt_g_imu;
        std::vector<Eigen::Vector3d> gt_angluar_vel;
        Eigen::Vector3d avgBg;
        avgBg.setZero();

        auto get_traj = [&](double timeStamp, GtData &rhs) -> bool {
            Eigen::map<double, GtData> gt_data = cam_gt;

            for (const auto &traj: gt_data) {
                if (std::abs((traj.first - timeStamp)) < 1e-3) {
                    rhs = traj.second;
                    return true;
                }
            }
            return false;
        };

        try {
            for (auto &t: idx_time) {
                GtData rhs;
                if (get_traj(t, rhs)) {
                    gt_pos.emplace_back(rhs.position);
                    gt_vel.emplace_back(rhs.velocity);
                    gt_rot.emplace_back(rhs.rotation.toRotationMatrix());
                    gt_g_imu.emplace_back(rhs.rotation.inverse() * G);

                    avgBg += rhs.bias_gyr;
                } else {
                    std::cout << "no gt pose,fail" << std::endl;
                    throw -1;
                }
            }
        } catch (...) {
            save_file << "time: " << fixed << idx_time[0] << " other_reason" << std::endl;
            save_file << "scale_error: " << "nan" << std::endl;
            save_file << "pose_error: " << "nan" << std::endl;
            save_file << "biasg_error: " << "nan" << std::endl;
            save_file << "velo_error: " << "nan" << std::endl;
            save_file << "gravity_error: " << "nan" << " " << "nan" << std::endl;
            save_file << "v0_error: " << "nan" << std::endl;
            LOG(INFO) << "---scale: ";
            std::cout << "time: " << fixed << idx_time[0] << std::endl;
            std::cout << "scale_error: " << 100 << std::endl;
            std::cout << "pose_error: " << 100 << std::endl;
            std::cout << "biasg_error: " << 100 << std::endl;
            std::cout << "velo_error: " << 100 << std::endl;
            std::cout << "rot_error: " << "nan" << std::endl;
            continue;
        }

        avgBg /= idx_time.size();

        double rot_rmse = 0;

        // rotation accuracy estimation
        for (int i = 0; i < idx_time.size() - 1; i++) {
            int j = i + 1;
            Eigen::Matrix3d rij_est = pDrtVioInit->rotation[i].transpose() * pDrtVioInit->rotation[j];
            Eigen::Matrix3d rij_gt = gt_rot[i].transpose() * gt_rot[j];
            Eigen::Quaterniond qij_est = Eigen::Quaterniond(rij_est);
            Eigen::Quaterniond qij_gt = Eigen::Quaterniond(rij_gt);
            double error =
                    std::acos(((qij_gt * qij_est.inverse()).toRotationMatrix().trace() - 1.0) / 2.0) * 180.0 / M_PI;
            rot_rmse += error * error;
        }
        rot_rmse /= (idx_time.size() - 1);
        rot_rmse = std::sqrt(rot_rmse);

        // translation accuracy estimation
        Eigen::Matrix<double, 3, Eigen::Dynamic> est_aligned_pose(3, idx_time.size());
        Eigen::Matrix<double, 3, Eigen::Dynamic> gt_aligned_pose(3, idx_time.size());

        for (int i = 0; i < idx_time.size(); i++) {
            est_aligned_pose(0, i) = pDrtVioInit->position[i](0);
            est_aligned_pose(1, i) = pDrtVioInit->position[i](1);
            est_aligned_pose(2, i) = pDrtVioInit->position[i](2);

            gt_aligned_pose(0, i) = gt_pos[i](0);
            gt_aligned_pose(1, i) = gt_pos[i](1);
            gt_aligned_pose(2, i) = gt_pos[i](2);
        }


        Eigen::Matrix4d Tts = Eigen::umeyama(est_aligned_pose, gt_aligned_pose, true);
        Eigen::Matrix3d cR = Tts.block<3, 3>(0, 0);
        Eigen::Vector3d t = Tts.block<3, 1>(0, 3);
        double s = cR.determinant();
        s = pow(s, 1.0 / 3);
        Eigen::Matrix3d R = cR / s;

        double pose_rmse = 0;
        for (int i = 0; i < idx_time.size(); i++) {
            Eigen::Vector3d target_pose = R * est_aligned_pose.col(i) + t;
            pose_rmse += (target_pose - gt_aligned_pose.col(i)).dot(target_pose - gt_aligned_pose.col(i));

        }
        pose_rmse /= idx_time.size();
        pose_rmse = std::sqrt(pose_rmse);

        std::cout << "vins sfm pose rmse: " << pose_rmse << std::endl;

        // gravity accuracy estimation
        double gravity_error =
                180. * std::acos(pDrtVioInit->gravity.normalized().dot(gt_g_imu[0].normalized())) / EIGEN_PI;

        // gyroscope bias accuracy estimation
        Eigen::Vector3d Bgs = pDrtVioInit->biasg;

        LOG(INFO) << "calculate bias: " << Bgs.x() << " " << Bgs.y() << " " << Bgs.z();
        LOG(INFO) << "gt bias: " << avgBg.x() << " " << avgBg.y() << " " << avgBg.z();

        const double scale_error = std::abs(s - 1.);
        const double gyro_bias_error = 100. * std::abs(Bgs.norm() - avgBg.norm()) / avgBg.norm();
        const double gyro_bias_error2 = 180. * std::acos(Bgs.normalized().dot(avgBg.normalized())) / EIGEN_PI;
        const double pose_error = pose_rmse;
        const double rot_error = rot_rmse;


        // velocity accuracy estimation
        double velo_norm_rmse = 0;
        double mean_velo = 0;
        for (int i = 0; i < idx_time.size(); i++) {
            velo_norm_rmse += (gt_vel[i].norm() - pDrtVioInit->velocity[i].norm()) *
                              (gt_vel[i].norm() - pDrtVioInit->velocity[i].norm());
            mean_velo += gt_vel[i].norm();
        }

        velo_norm_rmse /= idx_time.size();
        velo_norm_rmse = std::sqrt(velo_norm_rmse);
        mean_velo = mean_velo / idx_time.size();

        // the initial velocity accuracy estimation
        double v0_error = std::abs(gt_vel[0].norm() - pDrtVioInit->velocity[0].norm());

        std::cout << "integrate time: " << fixed << *idx_time.begin() << " " << *idx_time.rbegin() << " "
                  << *idx_time.rbegin() - *idx_time.begin() << std::endl;
        std::cout << "pose error: " << pose_error << " m" << std::endl;
        std::cout << "biasg error: " << gyro_bias_error << " %" << std::endl;
        std::cout << "gravity_error: " << gravity_error << std::endl;
        std::cout << "scale error: " << scale_error * 100 << " %" << std::endl;
        std::cout << "velo error: " << velo_norm_rmse << " m/s" << std::endl;
        std::cout << "v0_error: " << v0_error << std::endl;
        std::cout << "rot error: " << rot_error << std::endl;

        if (std::abs(s - 1) > 0.5 or std::abs(gravity_error) > 10) {
            LOG(INFO) << "===scale: " << s << " " << "gravity error: " << gravity_error;
            save_file << "time: " << fixed << idx_time[0] << " scale_gravity_fail" << std::endl;
            save_file << "scale_error: " << scale_error << std::endl;
            save_file << "pose_error: " << pose_error << std::endl;
            save_file << "biasg_error: " << gyro_bias_error << std::endl;
            save_file << "velo_error: " << velo_norm_rmse << std::endl;
            save_file << "gravity_error: " << gravity_error << " " << rot_error  << std::endl;
            save_file << "v0_error: " << v0_error << std::endl;
        } else {
            LOG(INFO) << "***scale: " << s << " " << "gravity error: " << gravity_error;
            save_file << "time: " << fixed << idx_time[0] << " good" << std::endl;
            save_file << "scale_error: " << scale_error << std::endl;
            save_file << "pose_error: " << pose_error << std::endl;
            save_file << "biasg_error: " << gyro_bias_error << std::endl;
            save_file << "velo_error: " << velo_norm_rmse << std::endl;
            save_file << "gravity_error: " << gravity_error << " " << rot_error << std::endl;
            save_file << "v0_error: " << v0_error << std::endl;
        }

    }

}
