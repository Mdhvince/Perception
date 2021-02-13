#include <iostream>
#include <array>
#include <cmath>

#include <eigen/Eigen/Dense>
#include <eigen/unsupported/Eigen/KroneckerProduct>
#include <opencv2/opencv.hpp>

using cv::ORB;
using cv::DescriptorMatcher;


Eigen::Vector3f line_from_points(Eigen::Vector3f &p1, Eigen::Vector3f &p2, bool normalize){
    // return line homogeneous coordinates (a b c) of equation of line ax + bx + c = 0;
    Eigen::Vector3f line = p1.cross(p2);

    if(normalize)
        line /= sqrtf(std::pow(line(0), 2) + std::pow(line(1), 2));
    return line;
}

Eigen::Vector3f point_from_lines(Eigen::Vector3f &l1, Eigen::Vector3f &l2, bool normalize){
    // return point homogeneous coordinates resulting of the intersection of the 2 lines
    Eigen::Vector3f point = l1.cross(l2);

    if(normalize) // to visualize
        point /= point(2);

    return point;
}

float angle_between_rays(Eigen::Vector3f &p_center, Eigen::Vector3f &p1, Eigen::Vector3f &p2){
    /* we have rays going from the projective center of the image to the 3D objects in the world
     X0 (Proj center) -> to X1 (3D point in the world)
     X0 (Proj center) -> to X2 (3D point in the world) */
    
    Eigen::Vector3f term1 = p1 - p_center;
    Eigen::Vector3f term2 = p2 - p_center; 
    float norm1 = sqrtf(std::pow(term1(0), 2) + std::pow(term1(1), 2) + std::pow(term1(2), 2));
    float norm2 = sqrtf(std::pow(term2(0), 2) + std::pow(term2(1), 2) + std::pow(term2(2), 2));
    float cos_angle = term1.dot(term2) / (norm1 * norm2);

    return std::acos(cos_angle);
}

bool point_on_line(Eigen::Vector3f &p, Eigen::Vector3f &line){
    bool lie_on = (line(0)*p(0) + line(1)*p(1) + line(2) == 0.0);
    return lie_on;
}

float distance_from_origin(Eigen::Vector3f &v){
    return v(2);
}

float direction_of_normal_line(Eigen::Vector3f &l){
    // rad
    float denom {sqrtf( std::pow(l(0), 2) + std::pow(l(1), 2) )};
    float theta = std::acos(l(0) / denom);
    return theta;
}


// Pose estimation (of the camera) : Extract R and t from H
void get_camera_position(Eigen::Matrix3f K, Eigen::Matrix3f H){

    Eigen::Matrix3f i_K = K.inverse();
    
    Eigen::Vector3f h1, h2, h3;
    h1 << i_K * H(Eigen::all, 0);
    h2 << i_K * H(Eigen::all, 1);
    h3 << i_K * H(Eigen::all, 2);

    float h1_norm = h1.norm();

    // find r1 r2 r3 that are the closest to h1 h2 h1xh2 : we just use SVD for this
    Eigen::Matrix3f h_bar;
    h_bar << h1, h2, h1.cross(h2); // hstack

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(h_bar, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto V_t = svd.matrixV().transpose();
    auto U = svd.matrixU();
    Eigen::Matrix3f R, R_bar;
    R_bar << 1, 0, 0,
             0, 1, 0,
             0, 0, (U*V_t).determinant();
    
    R = R_bar * V_t;
    Eigen::Vector3f t (h3 / h1_norm);

    // Get Camera center position P = K R [I  | -C] , with C = -inv(R) t
    Eigen::Vector3f C = -R.inverse() * t;

    // Eigen::Matrix<float, 3, 4> I_C;
    // I_C << 1, 0, 0, -C(0),
    //        0, 1, 0, -C(1),
    //        0, 0, 0, -C(2);
    
    // Eigen::Matrix<float, 3, 4> P = K * R * I_C;

}



int main(){

    Eigen::Vector3f p1(284., 63., 1.);
    Eigen::Vector3f p2(3., 8., 1.);
    Eigen::Vector3f p3(203., 12., 1.);

    Eigen::Vector3f line1 = line_from_points(p1, p2, false);
    Eigen::Vector3f line2 = line_from_points(p2, p3, false);
    Eigen::Vector3f point = point_from_lines(line1, line2, false);


    std::string impath1 {"/home/mdhvince/Coding/cpp/Perception/images/im1.jpg"};
    std::string impath2 {"/home/mdhvince/Coding/cpp/Perception/images/im2.jpg"};
    cv::Mat im1 = cv::imread(impath1, cv::IMREAD_COLOR);
    cv::Mat im2 = cv::imread(impath2, cv::IMREAD_COLOR);

    // FIND FUNDAMENTAL MATRIX THANKS TO THE 8 Points Algorithm
    // Get 8 matching points
    Eigen::Matrix<float, 8, 3> points_im1;
    Eigen::Matrix<float, 8, 3> points_im2;
    points_im1 << 198., 540., 1.,
                  198., 526., 1.,
                  205., 526., 1.,
                  316., 566., 1.,
                  344., 565., 1.,
                  104., 487., 1.,
                  102., 458., 1.,
                  424., 590., 1.;

    points_im2 << 224., 557., 1.,
                  224., 542., 1.,
                  233., 539., 1.,
                  406., 568., 1.,
                  427., 562., 1.,
                  15., 507., 1.,
                  11., 469., 1.,
                  519., 575., 1.;
    
    // In practice we should scale x and y coordinates for numerical stability
    // step 1: transform the pts so that the center of mass of all points are (0, 0)
    // step 2: scale the image so that x y coordinates are within [-1 1]
    
    /* Fundamental matrix equation:
    X'(rowformat) * F(3x3) * X"(columnformat) = 0
    To transform this equation to a least square problem, we need to transform matrix into vectors, using the kronecker product */
    Eigen::Matrix<float, 8, 9> points;

    for(int i {0}; i < points_im1.rows(); i++){
        points.row(i) = Eigen::kroneckerProduct(points_im1.row(i), points_im2.row(i));
    }

    /* now we have A(points)^t * F(reshaped in 1xN) = 0
    We use SVD to solve: Af = 0 */
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(points, Eigen::ComputeThinV);

    auto V = svd.matrixV();
    auto last_V = V.rightCols(1);
    
    Eigen::Matrix3f F_hat;
    F_hat << last_V(0), last_V(1), last_V(2),
             last_V(3), last_V(4), last_V(5),
             last_V(6), last_V(7), last_V(8);


    /*Now we manipulate F_hat a bit in order to force it to be rank 2 so we can have the true Fundamental Matrix F
    To do this, we use SVD on F_hat, we force the last singular value of D to be 0 and we compute U*D*V to retrieve F */
    Eigen::JacobiSVD<Eigen::MatrixXf> svd_F_hat(F_hat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto D_val = svd_F_hat.singularValues();
    Eigen::Matrix3f new_D;
    new_D << D_val(0), 0, 0,
             0, D_val(1), 0,
             0, 0, 0;
    Eigen::Matrix3f F_constraint = svd_F_hat.matrixU() * new_D * (svd_F_hat.matrixV()).transpose();
    Eigen::Matrix3f F = F_constraint / F_constraint.norm();
    
    // FIND FUNDAMENTAL MATRIX END




    // Find Epipole and epipolar lines from F
    Eigen::Vector3f ep_line0_im2 = F * points_im2.row(3).transpose();
    Eigen::Vector3f ep_line1_im2 = F * points_im2.row(4).transpose();
    Eigen::Vector3f epipol = point_from_lines(ep_line0_im2, ep_line1_im2, true);

    std::cout<<epipol<<std::endl;
    cv::Point pt1 = cv::Point(epipol(1), epipol(0)); // Y X
    cv::circle(im1, pt1, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_8);













    // FIND ESSENTIAL MATRIX (SAME BUT FOR CALIBRATED CAMERAS)
    // each point are in the camera coordinates system by applying X = K.inverse() * X; (calibrated points)
    // Eigen::Matrix3f K;
    // int i {0};
    // Eigen::Matrix<float, 8, 3> Kpoints_im1;
    // for(auto row: points_im1.rowwise()){
    //     Kpoints_im1.row(i) = K.inverse() * row;
    //     i++;
    // }
    // int j {0};
    // Eigen::Matrix<float, 8, 3> Kpoints_im2;
    // for(auto row: points_im2.rowwise()){
    //     Kpoints_im2.row(i) = K.inverse() * row;
    //     j++;
    // }

    /* then we execute the same step as to find the F_hat and we stop at E_hat.
    And to find the true E, we do the same as to find F but with the first Two Singular values set to 1 and the last to 0 */

    // FIND R and t vector between the 2 cameras from E
    // we can use recoverPose() from opencv


    // FIND ESSENTIAL MATRIX END - Or just use findEssentialMat() from OpenCV (only 5 points is enough)








    // for(auto row : points_im1.rowwise()){
    //     cv::Point pt1 = cv::Point(row(0), row(1));
    //     cv::circle(im1, pt1, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_8);
    // }
    // for(auto row : points_im2.rowwise()){
    //     cv::Point pt1 = cv::Point(row(0), row(1));
    //     cv::circle(im2, pt1, 3, cv::Scalar(255, 0, 255), -1, cv::LINE_8);
    // }

    // cv::Mat im1Gray, im2Gray;
    // cvtColor(resized1, im1Gray, cv::COLOR_BGR2GRAY);
    // cvtColor(resized2, im2Gray, cv::COLOR_BGR2GRAY);
    
    // std::vector<cv::KeyPoint> keypoints1, keypoints2;
    // cv::Mat descriptors1, descriptors2;
    // const int MAX_FEATURES {100};
    
    // // Detect ORB features and compute descriptors.
    // cv::Ptr<cv::Feature2D> orb = ORB::create(MAX_FEATURES);
    // orb->detectAndCompute(im1Gray, cv::Mat(), keypoints1, descriptors1);
    // orb->detectAndCompute(im2Gray, cv::Mat(), keypoints2, descriptors2);
    
    // // Match features.
    // std::vector<cv::DMatch> matches;
    // cv::Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    // matcher->match(descriptors1, descriptors2, matches, cv::Mat());
    
    // // Sort matches by score
    // std::sort(matches.begin(), matches.end());
    
    // // Draw top matches
    // cv::Mat imMatches;
    // drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);

    // cv::namedWindow("Image",cv::WINDOW_AUTOSIZE);
    // cv::imshow("Image", im1);

    // cv::namedWindow("Image2",cv::WINDOW_AUTOSIZE);
    // cv::imshow("Image2", im2);

    // cv::waitKey(0);


    std::cout<<"\n\n";
    return 0;
}