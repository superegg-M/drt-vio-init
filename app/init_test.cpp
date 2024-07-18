#include "featureTracker/featureTracker.h"
#include "featureTracker/parameters.h"
#include "IMU/imuPreintegrated.hpp"
#include "initMethod/drtVioInit.h"
#include "initMethod/drtLooselyCoupled.h"
#include "initMethod/drtTightlyCoupled.h"
#include "utils/eigenUtils.hpp"
#include "utils/ticToc.h"

#include <glog/logging.h>
#include <string>

using namespace std;
using namespace cv;
using namespace Eigen;

class Simulator
{
public:
    explicit Simulator(double dt, double w = 0.5, double r = 20.) : _dt(dt), _w(w), _r(r)
    {
        _ba.setZero();
        _bg.setZero();

        // landmarks生成
        double deg2rad = double(EIGEN_PI) / 180.;
        std::uniform_real_distribution<double> r_rand(0., 5.);
        std::uniform_real_distribution<double> z_rand(-5., 5.);
        for (int i = 0; i < 360; ++i)
        {
            double angle = double(i % 360) * deg2rad;
            double cos_ang = cos(angle);
            double sin_ang = sin(angle);
            // 轴向
            for (int j = 0; j < 5; ++j)
            {
                double l = r + double(j);
                //                double l = 0.5 * r + r_rand(_generator);
                for (int k = 0; k < 10; ++k)
                {
                    /*
                     * 把 p = (0, l, k), 旋转R
                     * 其中,
                     * R = [cos(theta) -sin(theta) 0
                     *      sin(theta) cos(theta) 0
                     *      0 0 1]
                     * */
                    landmarks[i][j][k] = {-l * cos_ang, -l * sin_ang, double(k) - 5.};
                    //                    landmarks[i][j][k] = {-l * cos_ang, -l * sin_ang, z_rand(_generator)};
                }
            }
            //            std::cout << "landmarks[i][j][k] = " << landmarks[i][0][0].transpose() << std::endl;
        }
    }

    void generate_data(unsigned int num_data)
    {
        _timestamp_buff.resize(num_data);
        _theta_buff.resize(num_data);
        _p_buff.resize(num_data);
        _v_buff.resize(num_data);
        _a_buff.resize(num_data);
        _w_buff.resize(num_data);
        for (unsigned int i = 0; i < num_data; ++i)
        {
            _timestamp_buff[i] = double(i) * _dt;

            _timestamp2_idx.emplace(_timestamp_buff[i], i);

            _theta_buff[i] = double(i) * _dt * _w;

            _p_buff[i].x() = _r * cos(_theta_buff[i]);
            _p_buff[i].y() = _r * sin(_theta_buff[i]);
            _p_buff[i].z() = 0.;

            _v_buff[i].x() = -_r * _w * sin(_theta_buff[i]);
            _v_buff[i].y() = _r * _w * cos(_theta_buff[i]);
            _v_buff[i].z() = 0.;

            _a_buff[i].x() = -_r * _w * _w + _ba.x();
            _a_buff[i].y() = 0. + _ba.y();
            _a_buff[i].z() = 9.8 + _ba.z();

            _w_buff[i].x() = 0. + _bg.x();
            _w_buff[i].y() = 0. + _bg.y();
            _w_buff[i].z() = _w + _bg.z();
        }
    }

    unsigned long get_landmark_id(unsigned int i, unsigned int j, unsigned int k)
    {
        return i + j * 1000 + k * 10000;
    }

    Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 7, 1>>>> get_landmarks_per_pose(double theta, const Matrix<double, 3, 1> &t_wi)
    {
        static double rad2deg = 180. / double(EIGEN_PI);
        Quaterniond q_wi{cos(0.5 * theta), 0., 0., sin(0.5 * theta)};

        Matrix<double, 3, 1> p_i, p_c;
        Matrix<double, 7, 1> f;
        Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 7, 1>>>> landmarks_map;

        int ang = (int(theta * rad2deg) + 360) % 360;
        //        ang /= 5;
        //        ang *= 5;
        for (int i = -10; i <= 10; i += 1)
        {
            int index = (ang + i + 360) % 360;
            for (int j = 0; j < 5; ++j)
            {
                for (int k = 0; k < 5; ++k)
                {
                    p_i = q_wi.inverse() * (landmarks[index][j][k] - t_wi);
                    p_c = q_ic.inverse() * (p_i - t_ic);
                    f << p_c.x() / p_c.z(), p_c.y() / p_c.z(), 1., (p_c.x() / p_c.z())*460+255, (p_c.y() / p_c.z())*460+255, 0., 0.;
                    landmarks_map[get_landmark_id(index, j, k)].emplace_back(0, f);
                    //                    std::cout << "p_c = " << p_c.transpose() << std::endl;
                }
            }
            //            if (i == 0) {
            //                std::cout << "p_i = " << p_i.transpose() << std::endl;
            //            }
        }

        return landmarks_map;
    }

public:
    vector<double> _timestamp_buff;
    vector<double> _theta_buff;
    vector<Matrix<double, 3, 1>> _p_buff;
    vector<Matrix<double, 3, 1>> _v_buff;
    vector<Matrix<double, 3, 1>> _a_buff;
    vector<Matrix<double, 3, 1>> _w_buff;

    std::map<double, int> _timestamp2_idx;

public:
    Matrix<double, 3, 1> _ba{0., 0., 0.};
    Matrix<double, 3, 1> _bg{0., 0., 0.};

public:
    double _dt;
    double _w;
    double _r;

public:
    Matrix<double, 3, 1> landmarks[360][5][10];
    std::default_random_engine _generator;

public:
    Quaterniond q_ic{cos(-0.5 * double(EIGEN_PI) * 0.5), 0., sin(-0.5 * double(EIGEN_PI) * 0.5), 0.};
    Matrix<double, 3, 1> t_ic{0., 0., 0.};
};

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: code" << " code type" << " data type" << "\n";
        return -1;
    }

    char *codeType = argv[1];
    char *dataType = argv[2];
    std::ofstream save_file("../result/" + string(codeType) + "_" + string(dataType) + ".txt");

    // IMU200Hz
    double dt = 0.005;
    unsigned int num_data = 6000 + 1;
    static Simulator simulator(dt);
    simulator.generate_data(num_data);

    readParameters("../config/simulator.yaml");
    double sf = std::sqrt(double(IMU_FREQ));

    for (int n = 0; n < num_data - 20 * 100; n += 20 * 10)
    {

        // 定义初始化指针
        DRT::drtVioInit::Ptr pDrtVioInit;
        // 设置初始化模式和外参
        if (string(codeType) == "drtTightly")
        {
            pDrtVioInit.reset(new DRT::drtTightlyCoupled(RIC[0], TIC[0]));
        }

        if (string(codeType) == "drtLoosely")
        {
            pDrtVioInit.reset(new DRT::drtLooselyCoupled(RIC[0], TIC[0]));
        }

        std::vector<int> idx;

        // // TODO：没明白为啥4Hz
        // // 40 4HZ, 0.25s
        // for (int j = n; j < 100 + n; j += 1)
        //     idx.push_back(j);

        // 图像10Hz，取100帧
        for (int j = n; j < 20 * 100 + n; j += 20)
            idx.push_back(j);

        double last_img_t_s, cur_img_t_s;
        bool first_img = true;
        bool init_feature = true;

        std::vector<double> idx_time;

        for (int i : idx)
        {

            cur_img_t_s = simulator._timestamp_buff[i];

            Eigen::aligned_map<int, Eigen::aligned_vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
                image;

            image = simulator.get_landmarks_per_pose(simulator._theta_buff[i], simulator._p_buff[i]);

            if (init_feature)
            {
                init_feature = false;
                continue;
            }

            // 检查视差是否满足要求，满足则添加该帧图像
            if (pDrtVioInit->addFeatureCheckParallax(cur_img_t_s, image, 0.0))
            {

                idx_time.push_back(cur_img_t_s);

                std::cout << "add image is: " << fixed << cur_img_t_s << " image number is: " << idx_time.size()
                          << std::endl;

                if (first_img)
                {
                    last_img_t_s = cur_img_t_s;
                    first_img = false;
                    continue;
                }

                auto GyroData = simulator._w_buff;
                auto AccelData = simulator._a_buff;

                std::vector<MotionData> imu_segment;

                for (size_t i = 0; i < GyroData.size(); i++)
                {
                    double timestamp = simulator._timestamp_buff[i];

                    MotionData imu_data;
                    imu_data.timestamp = timestamp;
                    imu_data.imu_acc = AccelData[i];
                    imu_data.imu_gyro = GyroData[i];

                    if (timestamp > last_img_t_s && timestamp <= cur_img_t_s)
                    {
                        imu_segment.push_back(imu_data);
                    }
                    if (timestamp > cur_img_t_s)
                    {
                        imu_segment.push_back(imu_data);
                        break;
                    }
                }

                vio::IMUBias bias;
                vio::IMUCalibParam
                    imu_calib(RIC[0], TIC[0], GYR_N * sf, ACC_N * sf, GYR_W / sf, ACC_W / sf);
                vio::IMUPreintegrated imu_preint(bias, &imu_calib, last_img_t_s, cur_img_t_s);

                int n = imu_segment.size() - 1;

                for (int i = 0; i < n; i++)
                {
                    double dt;
                    Eigen::Vector3d gyro;
                    Eigen::Vector3d acc;

                    if (i == 0 && i < (n - 1)) // [start_time, imu[0].time]
                    {
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tini = imu_segment[i].timestamp - last_img_t_s;
                        CHECK(tini >= 0);
                        acc = (imu_segment[i + 1].imu_acc + imu_segment[i].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tini / tab)) *
                              0.5f;
                        gyro = (imu_segment[i + 1].imu_gyro + imu_segment[i].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tini / tab)) *
                               0.5f;
                        dt = imu_segment[i + 1].timestamp - last_img_t_s;
                    }
                    else if (i < (n - 1)) // [imu[i].time, imu[i+1].time]
                    {
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc) * 0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro) * 0.5f;
                        dt = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                    }
                    else if (i > 0 && i == n - 1)
                    {
                        // std::cout << " n : " << i + 1 << " " << n << " " << imu_segment[i + 1].timestamp << std::endl;
                        float tab = imu_segment[i + 1].timestamp - imu_segment[i].timestamp;
                        float tend = imu_segment[i + 1].timestamp - cur_img_t_s;
                        CHECK(tend >= 0);
                        acc = (imu_segment[i].imu_acc + imu_segment[i + 1].imu_acc -
                               (imu_segment[i + 1].imu_acc - imu_segment[i].imu_acc) * (tend / tab)) *
                              0.5f;
                        gyro = (imu_segment[i].imu_gyro + imu_segment[i + 1].imu_gyro -
                                (imu_segment[i + 1].imu_gyro - imu_segment[i].imu_gyro) * (tend / tab)) *
                               0.5f;
                        dt = cur_img_t_s - imu_segment[i].timestamp;
                    }
                    else if (i == 0 && i == (n - 1))
                    {
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

        try {
            for (auto &t: idx_time) {
                gt_pos.emplace_back(simulator._p_buff[simulator._timestamp2_idx[t]]);
                gt_vel.emplace_back(simulator._v_buff[simulator._timestamp2_idx[t]]);
                Eigen::Quaterniond q_wi {cos(0.5 * simulator._theta_buff[simulator._timestamp2_idx[t]]), 0., 0., sin(0.5 * simulator._theta_buff[simulator._timestamp2_idx[t]])};
                gt_rot.emplace_back(q_wi.toRotationMatrix());
                gt_g_imu.emplace_back(q_wi.inverse() * G);

                avgBg += simulator._bg;

                std::cout << "gt geted" << std::endl;

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