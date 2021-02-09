#include <iostream>
#include <array>
#include <cmath>
#include <eigen/Eigen/Dense>


std::array< std::array<float, 3>, 3 > inv_3by3(std::array< std::array<float, 3>, 3 > &mat) {

    float a {mat.at(0).at(0)};
    float b {mat.at(0).at(1)};
    float c {mat.at(0).at(2)};

    float d {mat.at(1).at(0)};
    float e {mat.at(1).at(1)};
    float f {mat.at(1).at(2)};

    float g {mat.at(2).at(0)};
    float h {mat.at(2).at(1)};
    float i {mat.at(2).at(2)};

    float m11 { (e*i - f*h)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };
    float m12 { -(b*i - c*h)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };
    float m13 { (b*f - c*e)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };
    float m21 { -(d*i - f*g)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };
    float m22 { (a*i - c*g)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };
    float m23 { -(a*f - c*d)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };
    float m31 { (d*h - e*g)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };
    float m32 { -(a*h - b*g)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };
    float m33 { (a*e - b*d)/(a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g) };

    std::array< std::array<float, 3>, 3 > inv_mat {{
        {{ m11, m12, m13 }},
        {{ m21, m22, m23 }},
        {{ m31, m32, m33 }}

    }};

    return inv_mat;
}

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

int main(){

    Eigen::Vector3f p1(284., 63., 1.); //Vec3d are column=vector by default
    Eigen::Vector3f p2(3., 8., 1.);
    Eigen::Vector3f p3(203., 12., 1.);

    Eigen::Vector3f line1 = line_from_points(p1, p2, false);
    Eigen::Vector3f line2 = line_from_points(p2, p3, false);
    Eigen::Vector3f point = point_from_lines(line1, line2, false);


    // get calibration matrix K
    Eigen::Matrix3f K;
    // get H_tild = K [R t]
    // H = inv(k) * H_tild
    Eigen::Matrix3f H;
    H << .4430, .0037, -.1071,
         -.1153, .5216, .1506,
         .3096, .1875, .5944;

    // scaling factor
    float a = sqrtf( std::pow(H(0, 0), 2) + std::pow(H(1, 0), 2) + std::pow(H(2, 0), 2) );

    // translation vector
    Eigen::Vector3f t (H(0, 2) / a, H(1, 2) / a, H(2, 2) / a);
    Eigen::Vector3f r1 (H(0, 0) / a, H(1, 0) / a, H(2, 0) / a);
    Eigen::Vector3f r2 (H(0, 1) / a, H(1, 1) / a, H(2, 1) / a);
    Eigen::Vector3f r3 = r1.cross(r2);
    // rotation matrix
    Eigen::Matrix3f R;
    R <<  r1(0), r2(0), r3(0),
          r1(1), r2(1), r3(1),
          r1(2), r2(2), r3(2);

    // Get Camera center position P = K R [I  | -C] , with C = -inv(R) t
    Eigen::Vector3f C = -R.inverse() * t;

    Eigen::Matrix<float, 3, 4> I_C;
    I_C << 1, 0, 0, -C(0),
           0, 1, 0, -C(1),
           0, 0, 0, -C(2);
    
    Eigen::Matrix<float, 3, 4> P = K * R * I_C;



    std::cout<<"\n\n";
    return 0;
}