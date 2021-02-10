#include <iostream>
#include <array>
#include <cmath>

#include <eigen/Eigen/Dense>
#include <opencv2/opencv.hpp>


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


// homography estimation (At least 4 known points)
void update_mat_forH(Eigen::Matrix<float, Eigen::Dynamic, 9> &A, Eigen::Vector2f &p1_im, Eigen::Vector2f &p2_im){
    // A : Empty Matrix holder of size 2n x 9 : for 4 pts, A should be (8, 9)
    // this function update matrix A in a for loop before calling "estimate_h_transform()"

    float x1 = p1_im(0);
    float y1 = p1_im(1);
    float x2 = p2_im(0);
    float y2 = p2_im(1);

    Eigen::RowVectorXf ax(9);
    ax << -x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2;

    Eigen::RowVectorXf ay(9);
    ay << 0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2;

    A << ax,
         ay;
}

Eigen::Matrix3f estimate_h_transform(Eigen::Matrix<float, Eigen::Dynamic, 9> &A){

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::Matrix<float, Eigen::Dynamic, 9> V;
    V << svd.matrixV();

    Eigen::Map<Eigen::MatrixXf> h(V.leftCols(1).data(), 3,3); // last column of V Matrix reshaped

    Eigen::Matrix3f H;
    H << h.transpose();

    return H;
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
    cv::Mat im1 = imread(impath1, cv::IMREAD_COLOR);
    cv::Mat im2 = imread(impath2, cv::IMREAD_COLOR);
    cv::Mat resized1, resized2;
    cv::Size size(600,950); //Y, X
    
    cv::resize(im1,resized1, size);
    cv::resize(im2,resized2, size);

    cv::Mat stacked;
    cv::hconcat(resized1, resized2, stacked);


    cv::namedWindow("Image",cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", stacked);
    cv::waitKey(0);


    std::cout<<"\n\n";
    return 0;
}