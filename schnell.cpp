#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#if __cplusplus >= 202302L
#include <print>
#else
#include <fmt/core.h> 
#include <fmt/format.h>
#endif

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/types.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <boost/range/algorithm/transform.hpp>

#include <cxxopts.hpp>

#include <nlohmann/json.hpp>
#include <unistd.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#define BOOST_ALLOW_DEPRECATED_HEADERS

#include "cctag/Detection.hpp" // noisy fucking library

#undef BOOST_ALLOW_DEPRECATED_HEADERS
#pragma GCC diagnostic pop

using Eigen::Matrix3d;
using Eigen::Vector3;
using Eigen::Vector3d;
using Eigen::Vector4;
using Eigen::Vector4d;
using Vectors3d = std::vector<Vector3d>;
using nlohmann::json;
using std::ranges::transform;

#if __cplusplus >= 202302L
template<typename Type, int Size>
struct std::formatter<Eigen::Vector<Type, Size>>: std::formatter<std::string> {
    auto format(const Eigen::Vector<Type, Size>& v, std::format_context& ctx) const {
        return std::formatter<std::string>::format(
            std::accumulate(
                std::next(v.begin()),
                v.end(),
                std::format("[{}", v[0]),
                [](std::string a, const Type& x) { return std::format("{}, {}", std::move(a), x); }
            ) + ']',
            ctx
        );
    }
};
using std::print;
using std::format;
#else

// formater for jet data type
template <typename T, int N>
struct fmt::formatter<ceres::Jet<T, N>> : fmt::formatter<std::string> {
    auto format(const ceres::Jet<T, N>& jet, fmt::format_context& ctx) const {
        std::string v_str = "[";
        for (int i = 0; i < N; ++i) {
            v_str += fmt::format("{}", jet.v[i]);
            if (i < N - 1) v_str += ", ";
        }
        v_str += "]";
        return fmt::formatter<std::string>::format(fmt::format("Jet(a: {}, v: {})", jet.a, v_str), ctx);
    }
};

// // formater for Eigen::vector data type
// template<typename Type, int Size>
// struct fmt::formatter<Eigen::Vector<Type, Size>>: fmt::formatter<std::string> {
//     auto format(const Eigen::Vector<Type, Size>& v, fmt::format_context& ctx) const {
//         return fmt::formatter<std::string>::format(
//             std::accumulate(
//                 std::next(v.begin()),
//                 v.end(),
//                 fmt::format("[{}", v[0]),
//                 [](std::string a, const Type& x) { return fmt::format("{}, {}", std::move(a), x); }
//             ) + ']',
//             ctx
//         );
//     }
// };

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct fmt::formatter<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> : fmt::formatter<std::string> {
    static_assert(Cols == 1, "This formatter only supports column vectors.");

    auto format(const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& v, fmt::format_context& ctx) const {
        if (v.size() == 0)
            return fmt::formatter<std::string>::format("[]", ctx);

        return fmt::formatter<std::string>::format(
            std::accumulate(
                std::next(v.data()),  // skip first
                v.data() + v.size(),
                fmt::format("[{}", v[0]),
                [](std::string a, const Scalar& x) {
                    return fmt::format("{}, {}", std::move(a), x);
                }
            ) + ']',
            ctx
        );
    }
};

using fmt::print;
using fmt::format;
#endif

std::tuple<Vector3d, std::array<Vector3d, 3>> principal_components(const Vectors3d& vecs) {
    const Vector3d mean =
        std::accumulate(vecs.begin(), vecs.end(), Vector3d::Zero().eval()) / vecs.size();
    const Matrix3d cov = std::accumulate(
        vecs.begin(),
        vecs.end(),
        Matrix3d::Zero().eval(),
        [&mean](const Matrix3d& cov, const Vector3d& v) -> Matrix3d {
            Vector3d diff = v - mean;
            return cov + diff * diff.transpose();
        }
    );
    Matrix3d evecs = Eigen::SelfAdjointEigenSolver<Matrix3d>(cov).eigenvectors();
    return std::make_tuple(
        mean,
        std::array<Vector3d, 3> { evecs.col(0), evecs.col(1), evecs.col(2) }
    );
}

image_u8_t april_from_mat(const cv::Mat_<uint8_t>& m) {
    return image_u8_t { .width = m.cols, .height = m.rows, .stride = (int)m.step, .buf = m.data };
}

std::vector<apriltag_detection_t> dectvec_from_zarray(zarray_t*&& z) {
    std::vector<apriltag_detection_t> ret;
    apriltag_detection_t* d;

    int n = zarray_size(z);

    ret.reserve(n);
    for (int i = 0; i < n; ++i) {
        zarray_get(z, i, &d);
        ret.push_back(*d);
    }

    zarray_destroy(z);

    return ret;
}

void intersect_apriltag_dects(
    std::vector<apriltag_detection_t>& l,
    std::vector<apriltag_detection_t>& r,
    std::vector<cv::Vec2d>& lout,
    std::vector<cv::Vec2d>& rout
) {
    static const auto idcomp = [](const auto& l, const auto& r) { return l.id < r.id; };
    static const auto idmap = [](const auto& d) { return d.id; };

    std::vector<int> lids, rids;
    std::set<int> isect;

    std::ranges::sort(l, idcomp);
    std::ranges::sort(r, idcomp);

    transform(l, std::back_inserter(lids), idmap);
    transform(r, std::back_inserter(rids), idmap);

    std::ranges::set_intersection(lids, rids, std::inserter(isect, isect.begin()));

    auto fill_out = [&](const auto& ids, const auto& d, auto& out) {
        for (size_t i = 0; i < ids.size(); ++i)
            if (isect.contains(ids[i]))
                for (int j = 0; j < 4; ++j)
                    out.push_back({ d[i].p[j][0], d[i].p[j][1] }
                    ); // TODO reason about magic numbers
    };

    fill_out(lids, l, lout);
    fill_out(rids, r, rout);
}

void intersect_cctag_dects(
    boost::ptr_list<cctag::CCTag>& l,
    boost::ptr_list<cctag::CCTag>& r,
    std::vector<cv::Vec2d>& lout,
    std::vector<cv::Vec2d>& rout
) {
    static const auto idcomp = [](const auto& l, const auto& r) { return l.id() < r.id(); };
    static const auto idmap = [](const auto& d) { return d.id(); };

    std::vector<int> lids, rids;
    std::set<int> isect;

    l.sort(idcomp);
    r.sort(idcomp);

    boost::transform(l, std::back_inserter(lids), idmap);
    boost::transform(r, std::back_inserter(rids), idmap);

    std::ranges::set_intersection(lids, rids, std::inserter(isect, isect.begin()));

    auto fill_out = [&](const auto& ids, const auto& d, auto& out) {
        auto dit = d.cbegin();
        for (size_t i = 0; i < ids.size(); ++i, ++dit)
            if (isect.contains(ids[i]))
                out.push_back({ dit->x(), dit->y() });
    };

    fill_out(lids, l, lout);
    fill_out(rids, r, rout);
}

void detect_apriltags(
    const cv::Mat_<uint8_t>& limg,
    const cv::Mat_<uint8_t>& rimg,
    std::vector<cv::Vec2d>& ldects,
    std::vector<cv::Vec2d>& rdects
) {
    apriltag_detector_t* d = apriltag_detector_create();
    apriltag_family_t* f = tag36h11_create();
    apriltag_detector_add_family(d, f);

    image_u8_t lapr = april_from_mat(limg), rapr = april_from_mat(rimg);

    std::vector<apriltag_detection_t> ld = dectvec_from_zarray(apriltag_detector_detect(d, &lapr)),
                                      rd = dectvec_from_zarray(apriltag_detector_detect(d, &rapr));

    std::set<int> ids;

    apriltag_detector_remove_family(d, f);
    tag36h11_destroy(f);
    apriltag_detector_destroy(d);

    intersect_apriltag_dects(ld, rd, ldects, rdects);
}

void detect_cctags(
    const cv::Mat_<uint8_t>& limg,
    const cv::Mat_<uint8_t>& rimg,
    std::vector<cv::Vec2d>& ldects,
    std::vector<cv::Vec2d>& rdects
) {
    static const cctag::Parameters cctp { 4 };
    static const cctag::CCTagMarkersBank cctb { cctp._nCrowns };

    boost::ptr_list<cctag::CCTag> ld, rd;
    cctag::cctagDetection(ld, 0, 0, limg, cctp, cctb);
    cctag::cctagDetection(rd, 0, 0, rimg, cctp, cctb);

    intersect_cctag_dects(ld, rd, ldects, rdects);
}

void detect_sift(
    const cv::Mat_<uint8_t>& limg,
    const cv::Mat_<uint8_t>& rimg,
    const cv::Mat_<uint8_t>& lmask,
    const cv::Mat_<uint8_t>& rmask,
    std::vector<cv::Vec2d>& ldects,
    std::vector<cv::Vec2d>& rdects
) {
    std::vector<cv::KeyPoint> lkp, rkp;
    cv::Mat ldesc, rdesc;
    std::vector<std::vector<cv::DMatch>> matches;
    auto sift = cv::SIFT::create(0, 3, 0.05, 10, 1.6, true);
    auto matcher = cv::BFMatcher::create(cv::NORM_L2, true);

    sift->detectAndCompute(limg, lmask.empty() ? cv::noArray() : lmask, lkp, ldesc);
    sift->detectAndCompute(rimg, rmask.empty() ? cv::noArray() : rmask, rkp, rdesc);

    matcher->knnMatch(ldesc, rdesc, matches, 1);

    for (const auto& match: matches) {
        if (match.empty())
            continue;

        const auto& lpt = lkp[match[0].queryIdx].pt;
        const auto& rpt = rkp[match[0].trainIdx].pt;

        ldects.push_back({ lpt.x, lpt.y });
        rdects.push_back({ rpt.x, rpt.y });
    }

#if 0
    {
        cv::Mat drawnMatches;
        cv::drawMatches(limg, lkp, rimg, rkp, matches, drawnMatches);

        cv::imshow("matches", drawnMatches);
        cv::waitKey();
        cv::destroyAllWindows();
    }
#endif

    std::vector<char> mask;
    auto homography = cv::findHomography(ldects, rdects, cv::RANSAC, 3, mask);

    if (homography.empty())
        throw std::runtime_error("no homography!");

    for (ssize_t i = mask.size() - 1; i >= 0; --i) {
        if (!mask[i]) {
            ldects.erase(ldects.begin() + i);
            rdects.erase(rdects.begin() + i);
        }
    }
}

template<typename T>
constexpr T infinity = T(std::numeric_limits<double>::infinity());

template<typename T>
constexpr bool almost_zero(T x) {
    // https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#
    constexpr double atol = 1e-08;
    return ceres::abs(x) <= atol;
}

// Helper
template<typename T>
inline double get_value(const T& t) {
    return t;
}

template<typename T, int N>
inline T get_value(const ceres::Jet<T, N>& jet) {
    return jet.a;
}

template<typename T>
class Line {
    std::optional<Eigen::Matrix4<T>> _pluecker;

public:
    Vector3<T> dir, pt;

    Line(const Vector3<T>& direction, const Vector3<T>& point):
        dir(direction.normalized()),
        pt(point) {}

    Line(const Eigen::Matrix<T, 6, 1>& line):
        Line(
            Vector3<T>(line(0), line(1), line(2)),
            Vector3<T>(line(3), line(4), line(5))
        ) {}
    
    // Cast to another scalar type (e.g., double <-> Jet)
    template<typename OtherT>
    Line<OtherT> cast() const {
        return Line<OtherT>(
            dir.template cast<OtherT>(),
            pt.template cast<OtherT>()
        );
    }

    Vector3<T> distance_to(const Vector3<T>& other_pt) const {
        T proj_len = (other_pt - pt).dot(dir);
        Vector3<T> closest_pt = pt + proj_len * dir;

        return other_pt - closest_pt;
    }

    const Eigen::Matrix4<T> pluecker() {
        if (!_pluecker.has_value()) {
            Vector4<T> a = (pt + dir).homogeneous(), b = pt.homogeneous();

            _pluecker = a * b.transpose() - b * a.transpose();
        }

        return _pluecker.value();
    }

    Eigen::Matrix<T, 6, 1> getLine() const {
        return { dir.x(), dir.y(), dir.z(), pt.x(), pt.y(), pt.z()};
    }

    Eigen::Matrix<double, 6, 1> getLineJet() const {
        Eigen::Matrix<double, 6, 1> out;
        out <<
            get_value(dir.x()), get_value(dir.y()), get_value(dir.z()),
            get_value(pt.x()), get_value(pt.y()), get_value(pt.z());
        return out;
    }

};

template<typename T>
struct SinusoidalWaveSurface {

    // p(x,y) = origin ​+ u⋅x + v⋅y + A⋅sin(kx + ly + phase) ⋅ n
    Vector3<T> origin;    // Reference point on the surface  // mean of pca for first guess
    Vector3<T> u, v;      // Tangent vectors defining the local x, y direction
    Vector3<T> normal;    // Direction of wave displacement
    T amplitude;          // A
    T frequency;          // k, affects wavelength
    T phase;              // φ

    // init as zero matrix if no variable given
    SinusoidalWaveSurface(): origin(Vector3<T>::Zero()), u(Vector3<T>::Zero()), v(Vector3<T>::Zero()),
                             normal(Vector3<T>::Zero()), amplitude(T(0)), frequency(T(0)), phase(T(0)) {}

    // Constructor for given values
    SinusoidalWaveSurface(const Vector3<T>& origin,
                          const Vector3<T>& u,
                          const Vector3<T>& v,
                          const Vector3<T>& normal,
                          T amplitude,
                          T frequency,
                          T phase = T(0))
        : origin(origin),
          u(u.normalized()),
          v(v.normalized()),
          normal(normal.normalized()),
          amplitude(amplitude),
          frequency(frequency),
          phase(phase) {}
    
    // Constructor for direction of wave + normal + sin wave params
    SinusoidalWaveSurface(const Vector3<T>& origin,
                          const Vector3<T>& u,
                          const Vector3<T>& normal,
                          T amplitude,
                          T frequency,
                          T phase = T(0))
        : origin(origin),
        u(u.normalized()), v(u.cross(normal).normalized()),
        normal(normal.normalized()),
        amplitude(amplitude),
        frequency(frequency),
        phase(phase) {}

    SinusoidalWaveSurface(const Eigen::Matrix<T, 15, 1>& sin_params):
        SinusoidalWaveSurface(
            Vector3<T>(sin_params(0), sin_params(1), sin_params(2)),
            Vector3<T>(sin_params(3), sin_params(4), sin_params(5)),
            Vector3<T>(sin_params(6), sin_params(7), sin_params(8)),
            Vector3<T>(sin_params(9), sin_params(10), sin_params(11)),
            sin_params(12),
            sin_params(13),
            sin_params(14)
        ) {}

    SinusoidalWaveSurface(T ox, T oy, T oz, T ux, T uy, T uz, T nx, T ny, T nz, T amplitude, T frequency, T phase):
        SinusoidalWaveSurface(
            Vector3<T>(ox, oy, oz),
            Vector3<T>(ux, uy, uz),
            Vector3<T>(nx, ny, nz),
            amplitude,
            frequency,
            phase
        ) {}

    SinusoidalWaveSurface(const std::vector<T>& sin_params):
        SinusoidalWaveSurface(
            Vector3<T>(sin_params.at(0), sin_params.at(1), sin_params.at(2)),
            Vector3<T>(sin_params.at(3), sin_params.at(4), sin_params.at(5)),
            Vector3<T>(sin_params.at(6), sin_params.at(7), sin_params.at(8)),
            sin_params.at(9),
            sin_params.at(10),
            sin_params.at(11)
        ) {}
    
    SinusoidalWaveSurface(const T* sin_params):
        SinusoidalWaveSurface(
            Vector3<T>(sin_params[0], sin_params[1], sin_params[2]),
            Vector3<T>(sin_params[3], sin_params[4], sin_params[5]),
            Vector3<T>(sin_params[6], sin_params[7], sin_params[8]),
            sin_params[9],
            sin_params[10],
            sin_params[11]
        ) {}
    
    template<typename OtherT>
    SinusoidalWaveSurface<OtherT> cast() const {
        return SinusoidalWaveSurface<OtherT>(
            origin.template cast<OtherT>(),
            u.template cast<OtherT>(),
            v.template cast<OtherT>(),
            normal.template cast<OtherT>(),
            static_cast<OtherT>(amplitude),
            static_cast<OtherT>(frequency),
            static_cast<OtherT>(phase)
        );
    }

    Eigen::Matrix<T, 15, 1> wave_param() const {
        return Eigen::Matrix<T, 15, 1>{
            origin.x(), origin.y(), origin.z(),
            u.x(), u.y(), u.z(),
            v.x(), v.y(), v.z(),
            normal.x(), normal.y(), normal.z(),
            amplitude, frequency, phase
        };
    }

    Eigen::Matrix<double, 15, 1> jet_param() const {
        Eigen::Matrix<double, 15, 1> out;
        out << 
            get_value(origin.x()), get_value(origin.y()), get_value(origin.z()),
            get_value(u.x()), get_value(u.y()), get_value(u.z()),
            get_value(v.x()), get_value(v.y()), get_value(v.z()),
            get_value(normal.x()), get_value(normal.y()), get_value(normal.z()),
            get_value(amplitude), get_value(frequency), get_value(phase);
        return out;
    }

    template<typename J>
    Vector3<T> evaluate(J x, J y) const {

        J wave = amplitude * ceres::sin(frequency * x + phase);

        return origin + x * u + y * v + wave * normal;
    }

    // get normal at a point (approximation)
    Vector3<T> normal_at(T x, T y, T dx = T(1e-4)) const {
        Vector3<T> p = evaluate(x, y);
        Vector3<T> px = evaluate(x + dx, y);
        Vector3<T> py = evaluate(x, y + dx);
        return (px - p).cross(py - p).normalized();
    }

    Vector3<T> refract(const Line<T>& line, bool backwards = false) const {
        double r = 1.33; // air -> water
        Vector3<T> n = normal_at(line.pt.x(), line.pt.y()); // get aproximate norm on the sin surface

        if (backwards) {
            n = n * -1.0;
        } else
            r = 1.0 / r;

        T c = -n.dot(line.dir);
        return r * line.dir + n * (r * c - ceres::sqrt(1.0 - r * r * (1.0 - c * c)));
    }
};

// Residual for the intersection of a line with a sinusoidal wave surface
template<typename T>
struct IntersectionResidual {
    Line<T> line_d;
    SinusoidalWaveSurface<T> surface_d;

    IntersectionResidual(const Line<double>& l, const SinusoidalWaveSurface<double>& s)
        :line_d(l), surface_d(s) {}

    template <typename J>
    bool operator()(const J* const xy, J* residuals) const {
        J x = xy[0];
        J y = xy[1];

        // Convert stored double-typed objects to Jet-typed
        Line<J> line = line_d.template cast<J>();
        SinusoidalWaveSurface<J> surface = surface_d.template cast<J>();

        // Calculate surface point (Evaluate the wave surface at (x, y))
        Vector3<J> p = surface.evaluate(x, y);
        Vector3<J> o = line.pt;               // Point on line
        Vector3<J> d = line.dir;              // Direction of line

        // Project p onto the line direction to find the closest point
        J t = (p - o).dot(d);
        Vector3<J> proj = o + t * d;

        // Calculate difference between point on the surface and the projection
        Vector3<J> diff = p - proj;

        // Set residuals (difference in 3D space)
        residuals[0] = diff.x();
        residuals[1] = diff.y();
        residuals[2] = diff.z();

        return true;
    }
};

/**
 * @brief Computes the intersection point of a line with a sinusoidal wave surface.
 * 
 * @param line The line object represented as a parameterized line in 3D space.
 * @param surface The sinusoidal wave surface object to intersect with.
 * @param isec Output parameter to store the computed intersection point in 3D space.
 * @return true If the intersection is successfully computed and within a reasonable distance.
 * @return false If the intersection is too far away or the computation fails.
 */
template<typename T>
bool intersectWithSinPlane(Line<T> line, SinusoidalWaveSurface<T> surface, Vector3<T>& isec) {

    // since only solution with double:
    const SinusoidalWaveSurface<double> tempplane(surface.jet_param());
    const Line<double> templine(line.getLineJet());

    // Initial guess for (x, y) parameters to evaluate on the surface
    double xy[2] = {0.0, 0.0};

    ceres::Problem problem;

    // Add the residual block
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<IntersectionResidual<double>, 3, 2>(
            new IntersectionResidual<double>{templine, tempplane},
            ceres::Ownership::TAKE_OWNERSHIP
        ),
        nullptr,  // no loss function
        xy        // the (x, y) on the surface
    );

    // TODO: see if better parameter can be chosen
    ceres::Solver::Options options;
    options.minimizer_type = ceres::MinimizerType::TRUST_REGION;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    options.minimizer_progress_to_stdout = true; 
    options.logging_type = ceres::SILENT;
    options.use_explicit_schur_complement = true; 
    options.update_state_every_iteration = true;
    options.function_tolerance = 1e-30;
    options.gradient_tolerance = 1e-30;
    options.parameter_tolerance = 1e-30;
    options.max_num_iterations = 100;
    options.num_threads = sysconf(_SC_NPROCESSORS_ONLN);


    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
   
    // print("isec sum: {}\n", summary.FullReport());

    // Get the resulting point on the surface
    Vector3<double> intersection = tempplane.evaluate(xy[0], xy[1]);

    isec = intersection.template cast<T>();

    return true;
}

template<typename T>
struct Plane {
    Vector3<T> abc;
    T d;

    Plane(): abc(Vector3<T>::Zero()) {}

    // costroctur for Vector3D
    Plane(const Vector3<T>& perpvec, const Vector3<T>& point) {
        // a(x - px) + b(y - py) + c(z - pz) = 0
        // d = -a*px - b*py - c*pz
        // ax + by + cz + (-a*px -b*py -c*pz) = 0
        abc = perpvec.normalized();
        d = point.dot(-abc);
    }

    // costructor for std::vector
    Plane(const std::vector<T>& perpvec, const std::vector<T>& point):
        Plane(
            Vector3<T> { perpvec.at(0), perpvec.at(1), perpvec.at(2) },
            Vector3<T> { point.at(0), point.at(1), point.at(2) }
        ) {}
    
    // constructor for variables
    Plane(T a, T b, T c, T d): abc(a, b, c), d(d) {
        abc.normalize();
    }

    // from a pointer
    Plane(const T* abcd): Plane(abcd[0], abcd[1], abcd[2], abcd[3]) {}

    Plane(const std::vector<T>& abcd): Plane(abcd.at(0), abcd.at(1), abcd.at(2), abcd.at(3)) {}

    // return the plane as a 4D vector
    Vector4<T> abcd() const {
        return { abc.x(), abc.y(), abc.z(), d };
    }

    // NOTE: not used
    Vector3<T> some_point() const {
        Vector3<T> ret;
        size_t maxidx;

        abc.head(3).cwiseAbs().maxCoeff(&maxidx);

        double max = abc[maxidx];

        if (almost_zero(max))
            print(stderr, "Almost zero maximum coefficient\n");

        switch (maxidx) {
            case 0: // a
                ret.y() = ret.z() = 0.0;
                ret.x() = -d / max;
                break;
            case 1: // b
                ret.x() = ret.z() = 0.0;
                ret.y() = -d / max;
                break;
            case 2: // c
                ret.x() = ret.y() = 0.0;
                ret.z() = -d / max;
                break;
            default:
                __builtin_unreachable();
        }

        return ret;
    }

    Vector4<T> intersect_with_(Line<T>& line) const {
        return line.pluecker().transpose() * abcd();
    }

    bool intersect_with(Line<T>& line, Vector3<T>& intersection) const {
        Vector4<T> hom = intersect_with_(line);
        if (almost_zero(hom.w()))
            return false;

        intersection = hom.hnormalized();
        return true;
    }

    Vector3<T> refract(const Line<T>& line, bool backwards = false) const {
        double r = 1.33; // air -> water
        Vector3<T> n = abc;

        // TODO make sense of this

        if (backwards) {
            n = n * -1.0;
        } else
            r = 1.0 / r;

        T c = -n.dot(line.dir);
        return r * line.dir + n * (r * c - ceres::sqrt(1.0 - r * r * (1.0 - c * c)));
    }
};

// NOTE: not used?
void to_json(nlohmann::json& j, const Plane<double>& p) {
    j = { { "pt", p.some_point() }, { "abcd", p.abcd() } };
}

template<typename T>
bool forward_refract_estimate(
    const Vector3<T>& pt,
    const Vector3<T>& T0,
    const Vector3<T>& T1,
    const Plane<T>& plane,
    Vector3<T>& lestimate,
    Vector3<T>& restimate,
    Vector3<T>& lisect,
    Vector3<T>& risect)
    {

    Line<T> lline(pt - T0, T0), rline(pt - T1, T1);

    if (!plane.intersect_with(lline, lisect))
        return false;
    if (!plane.intersect_with(rline, risect))
        return false;

    Vector3<T> lrefr = plane.refract(lline), rrefr = plane.refract(rline);

    Plane rrefrplane(rline.dir.cross(rrefr), risect), lrefrplane(lline.dir.cross(lrefr), lisect);

    Line lrefrline(lrefr, lisect), rrefrline(rrefr, risect);

    if (!rrefrplane.intersect_with(lrefrline, lestimate))
        return false;
    if (!lrefrplane.intersect_with(rrefrline, restimate))
        return false;

    return true;
}

template<typename T>
bool forward_refract_estimate_sin(
    const Vector3<T>& pt,
    const Vector3<T>& T0,
    const Vector3<T>& T1,
    const SinusoidalWaveSurface<T>& plane,
    Vector3<T>& lestimate,
    Vector3<T>& restimate,
    Vector3<T>& lisect,
    Vector3<T>& risect)
    {

    Line<T> lline(pt - T0, T0), rline(pt - T1, T1);

    if (!intersectWithSinPlane(lline, plane, lisect))
        return false;
    if (!intersectWithSinPlane(rline, plane, risect))
        return false;

    // print("lisect: {}\n", lisect);  
    // print("lline dir: {}\n", lline.dir);
    // print("lline origin: {}\n", lline.pt);
    // print("plane: {}\n", plane.wave_param());

    Vector3<T> lrefr = plane.refract(lline), rrefr = plane.refract(rline);

    Plane rrefrplane(rline.dir.cross(rrefr), risect), lrefrplane(lline.dir.cross(lrefr), lisect);

    Line lrefrline(lrefr, lisect), rrefrline(rrefr, risect);

    if (!rrefrplane.intersect_with(lrefrline, lestimate))
        return false;
    if (!lrefrplane.intersect_with(rrefrline, restimate))
        return false;

    return true;
}

template<typename T>
void back_refract(
    const Vector3<T>& estimate,
    const Vector3<T>& isect,
    const Vector3<T>& baseline,
    const Plane<T>& someplane,
    Vector3<T>& distance,
    Vector3<T>* backrefraction
) {
    static thread_local Vector3<T> backrefraction_;
    if (!backrefraction)
        backrefraction = &backrefraction_;

    *backrefraction = someplane.refract(Line<T> { isect - estimate, estimate }, true);

    distance = Line(*backrefraction, isect).distance_to(baseline);
}

template<typename T>
void back_refract_sin(
    const Vector3<T>& estimate,
    const Vector3<T>& isect,
    const Vector3<T>& baseline,
    const SinusoidalWaveSurface<T>& someplane,
    Vector3<T>& distance,
    Vector3<T>* backrefraction
) {
    static thread_local Vector3<T> backrefraction_;
    if (!backrefraction)
        backrefraction = &backrefraction_;

    *backrefraction = someplane.refract(Line<T> { isect - estimate, estimate }, true);

    distance = Line(*backrefraction, isect).distance_to(baseline);
}

struct MyCostFunctor {
    const Vector3d T0, T1;
    const Vector3d scene;

    MyCostFunctor(const Vector3d& warped, const Vector3d& T0, const Vector3d& T1):
        T0(T0),
        T1(T1),
        scene(warped) {}
};

// cost for plane waves
struct EstimatedDistanceCostFunctor: public MyCostFunctor {
    using MyCostFunctor::MyCostFunctor;

    template<typename T>
    bool operator()(const T* const abcd, T* residuals) const {
        
        const Plane someplane(abcd);

        Vector3<T> _lestimates, _restimates, _lisects, _risects, _back;

        bool ok = forward_refract_estimate<T>(
            scene.cast<T>(),
            T0.cast<T>(),
            T1.cast<T>(),
            someplane,
            _lestimates,
            _restimates,
            _lisects,
            _risects
        );

        if (!ok)
            return false;

        Vector3<T> distance = _lestimates - _restimates;

        residuals[0] = distance.x();
        residuals[1] = distance.y();
        residuals[2] = distance.z();

        // fmt::print("Res est distance: {}\n", distance);

        return true;
    }
};

// cost function for sin waves
struct EstimatedDistanceCostFunctorSinWave: public MyCostFunctor {
    using MyCostFunctor::MyCostFunctor;

    template<typename T>
    bool operator()(const T* const sin_params, T* residuals) const {
        
        const SinusoidalWaveSurface someplane(sin_params);

        Vector3<T> _lestimates, _restimates, _lisects, _risects, _back;

        bool ok = forward_refract_estimate_sin<T>(
            scene.cast<T>(),
            T0.cast<T>(),
            T1.cast<T>(),
            someplane,
            _lestimates,
            _restimates,
            _lisects,
            _risects
        );

        if (!ok)
            return false;

        Vector3<T> distance = _lestimates - _restimates;

        residuals[0] = distance.x();
        residuals[1] = distance.y();
        residuals[2] = distance.z();

        // fmt::print("Res est distance: {}\n", distance);

        return true;
    }
};

template<bool LEFT>
struct BackrefractionCostFunctor: public MyCostFunctor {
    using MyCostFunctor::MyCostFunctor;
    template<typename T>
    bool operator()(const T* const abcd, T* residuals) const {

        const Plane someplane(abcd);

        Vector3<T> _lestimates, _restimates, _lisects, _risects, _back;

        bool ok = forward_refract_estimate<T>(
            scene.cast<T>(),
            T0.cast<T>(),
            T1.cast<T>(),
            someplane,
            _lestimates,
            _restimates,
            _lisects,
            _risects
        );

        if (!ok)
            return false;

        if constexpr (LEFT)
            back_refract<T>(_lestimates, _risects, T0.cast<T>(), someplane, _back, nullptr);
        else
            back_refract<T>(_restimates, _lisects, T1.cast<T>(), someplane, _back, nullptr);

        residuals[0] = _back.x();
        residuals[1] = _back.y();
        residuals[2] = _back.z();

        // fmt::print("Res back: {}{}{}\n", _back.x(), _back.y(), _back.z());

        return true;
    }

};

template<bool LEFT>
struct BackrefractionCostFunctorSin: public MyCostFunctor {
    using MyCostFunctor::MyCostFunctor;
    template<typename T>
    bool operator()(const T* const sin_param, T* residuals) const {

        const SinusoidalWaveSurface someplane(sin_param);

        Vector3<T> _lestimates, _restimates, _lisects, _risects, _back;

        bool ok = forward_refract_estimate_sin<T>(
            scene.cast<T>(),
            T0.cast<T>(),
            T1.cast<T>(),
            someplane,
            _lestimates,
            _restimates,
            _lisects,
            _risects
        );

        if (!ok)
            return false;

        if constexpr (LEFT)
            back_refract_sin<T>(_lestimates, _risects, T0.cast<T>(), someplane, _back, nullptr);
        else
            back_refract_sin<T>(_restimates, _lisects, T1.cast<T>(), someplane, _back, nullptr);

        residuals[0] = _back.x();
        residuals[1] = _back.y();
        residuals[2] = _back.z();

        // fmt::print("Res back: {}{}{}\n", _back.x(), _back.y(), _back.z());

        return true;
    }

};

enum class DetectionType { APRIL, CCTAG, SIFT };

constexpr auto camidxs = { 0, 1, 2, 3 };
struct Combo {
    const std::tuple<int, int> idxs;
    const cv::Mat i1, i2, m1, m2;
    cv::Mat d1, d2;
    cv::Matx33d K1, K2, R;
    cv::Matx31d T;
    cv::Matx34d P1, P2;
    Eigen::Matrix4d RefTrans;

    Combo() {}

    Combo(int idx1, int idx2, const std::filesystem::path& datapath):
        idxs(std::make_tuple(idx1, idx2)),
        i1(cv::imread(datapath / format("{}.png", idx1), cv::IMREAD_GRAYSCALE)),
        i2(cv::imread(datapath / format("{}.png", idx2), cv::IMREAD_GRAYSCALE)),
        m1(cv::imread(datapath / format("{}_mask.png", idx1), cv::IMREAD_GRAYSCALE)),
        m2(cv::imread(datapath / format("{}_mask.png", idx2), cv::IMREAD_GRAYSCALE)),
        RefTrans(Eigen::Matrix4d::Identity()) {
        cv::FileStorage fs(
            datapath / format("{}-to-{}.json", idx1, idx2),
            cv::FileStorage::READ
        );

        fs["K1"] >> K1;
        fs["K2"] >> K2;
        fs["d1"] >> d1;
        fs["d2"] >> d2;
        fs["R"] >> R;
        fs["T"] >> T;

        cv::hconcat(K1, cv::Vec3d::zeros(), P1);
        {
            cv::Matx34d RT;
            cv::hconcat(R, T, RT);
            P2 = K2 * RT;
        }
    }

    void set_reference_frame(const cv::Matx33d& R, const cv::Matx31d& T) {
        cv::Matx34d upper;
        cv::Matx44d refTrans;
        cv::hconcat(R, T, upper);
        cv::vconcat(upper, cv::Matx14d(0, 0, 0, 1), refTrans);
        cv::cv2eigen(refTrans.inv(), RefTrans);
    }

    std::tuple<Vector3d, Vector3d> get_baseline_in_reference_frame() const {
        cv::Matx31d cv_baseline_local = -R.t() * T;
        Vector3d eigen_baseline_local;

        cv::cv2eigen(cv_baseline_local, eigen_baseline_local);
        return std::make_tuple(
            (RefTrans * Vector3d::Zero().homogeneous()).hnormalized(),
            (RefTrans * eigen_baseline_local.homogeneous()).hnormalized()
        );
    }

    Vectors3d triangulate_into_referece_frame(DetectionType dtype) const {
        std::vector<cv::Vec2d> dect1_distorted, dect1, dect2_distorted, dect2;
        cv::Mat points4d;
        Vectors3d ret;

        switch (dtype) {
            case DetectionType::APRIL:
                detect_apriltags(i1, i2, dect1_distorted, dect2_distorted);
                break;
            case DetectionType::CCTAG:
                detect_cctags(i1, i2, dect1_distorted, dect2_distorted);
                break;
            case DetectionType::SIFT:
                detect_sift(i1, i2, m1, m2, dect1_distorted, dect2_distorted);
                break;
        }

        cv::undistortImagePoints(dect1_distorted, dect1, K1, d1);
        cv::undistortImagePoints(dect2_distorted, dect2, K2, d2);

        cv::triangulatePoints(P1, P2, dect1, dect2, points4d);

        for (int col = 0; col < points4d.cols; ++col) {
            cv::Vec4f hom_pt = points4d.col(col);
            auto [x, y, z, w] = hom_pt.val;
            // TODO
            ret.push_back((RefTrans * Vector4d { x, y, z, w }).hnormalized());
        }

        return ret;
    }
};

class Recorder: public ceres::IterationCallback {
private:
    std::vector<Plane<double>> _steps;
    const double* const abcd;

public:
    Recorder(const double* abcd): abcd(abcd) {}
    ceres::CallbackReturnType operator()(const ceres::IterationSummary& _ [[maybe_unused]]
    ) override {
        _steps.push_back({ abcd });
        return ceres::CallbackReturnType::SOLVER_CONTINUE;
    }
    auto consume() {
        return std::move(_steps);
    }
};

class RecorderSin: public ceres::IterationCallback {
    private:
        std::vector<SinusoidalWaveSurface<double>> _steps;
        const double* const sin_params;
    
    public:
        RecorderSin(const double* sin_params): sin_params(sin_params) {}
        ceres::CallbackReturnType operator()(const ceres::IterationSummary& _ [[maybe_unused]]
        ) override {
            _steps.push_back({ sin_params });
            return ceres::CallbackReturnType::SOLVER_CONTINUE;
        }
        auto consume() {
            return std::move(_steps);
        }
    };

template<typename TVal>
using StereoMap = std::map<std::tuple<int, int>, TVal>;

int main(int argc, const char** argv) {
    
    // clang-format off
    cxxopts::Options options("schnell");

    // possible argument passde with --argument not only argument
    options.add_options()
        ("datapath", "path to data", cxxopts::value<std::string>())
        ("abcd", "initial plane values", cxxopts::value<std::vector<double>>())
        ("point", "xyz point on plane", cxxopts::value<std::vector<double>>())
        ("perpvec", "xyz components of vector perpendicular to plane", cxxopts::value<std::vector<double>>())
        ("sift", "enable sift detection")
        ("solve", "runs solver")
        ("sin", "enable sinusoidal wave surface")

        //("huber", "huber loss coefficient", cxxopts::value<double>()->default_value("0.1"))
        ("lone", "softlone loss coefficient", cxxopts::value<double>()->default_value("0.04"));

    options.parse_positional({"datapath"});

    const auto args = options.parse(argc, argv);
    
    // clang-format on

    bool solve = args.count("solve");
    bool sin = args.count("sin");
    DetectionType detection_type = args.count("sift") ? DetectionType::SIFT : DetectionType::APRIL;
    std::string datapath = args["datapath"].as<std::string>();

    print(stderr, "data path : {}\n", datapath);
    print(stderr, "type of detection: {}\n", args.count("sift") ? "sift" : "apriltag");
    print(stderr, "lone: {}\n", args.count("lone") ? args["lone"].as<double>() : 0.01);
    print(stderr, "sin: {}\n", args.count("sin") ? "optimizing sin plane" : "optimizing flat plane");

    StereoMap<Combo> combos;
    for (size_t i = 0; i < camidxs.size(); ++i) {
        for (size_t j = i + 1; j < camidxs.size(); ++j) {
            Combo combo(i, j, datapath);
            switch (i) {
                case 0:
                    break;
                case 1: {
                    auto reference = combos[std::make_tuple(0, 1)];
                    combo.set_reference_frame(reference.R, reference.T);
                } break;
                case 2: {
                    auto reference = combos[std::make_tuple(0, 2)];
                    combo.set_reference_frame(reference.R, reference.T);
                } break;
                default:
                    __builtin_unreachable();
            }
            combos.insert(std::make_pair(std::make_tuple(i, j), std::move(combo)));
        }
    }

    // get how to estimate the fisrt guess of the plane -> if no argument given pca of the plane
    bool guess_plane = !(args.count("abcd") || (args.count("perpvec") && args.count("point")));
    
    // init the plane
    Plane<double> someplane;
    SinusoidalWaveSurface<double> somesinplane;
    StereoMap<Vectors3d> triangulations;
    StereoMap<std::tuple<Vector3d, Vector3d>> camera_positions;

    for (const auto& [key, combo]: combos) {
        camera_positions[key] = combo.get_baseline_in_reference_frame();
        triangulations[key] = combo.triangulate_into_referece_frame(detection_type);
    }

    if (guess_plane) {
        Vectors3d pattern_center_vectors;

        // TODO see if it the right assigned -> works for tested case should be the biggest eigenvalue first
        Vectors3d pattern_evecs_Z;
        Vectors3d pattern_evecs_Y;
        Vectors3d pattern_evecs_X;

        for (const auto& [key, combo]: combos) {
            const auto& [T0, T1] = camera_positions[key];
            const auto [mean, evecs] = principal_components(triangulations[key]);
            pattern_center_vectors.push_back(mean - T0);
            pattern_center_vectors.push_back(mean - T1);
            pattern_evecs_Z.push_back(evecs[0]);
            pattern_evecs_Y.push_back(evecs[1]);
            pattern_evecs_X.push_back(evecs[2]);
        }

        Vector3d mean_pcv = std::accumulate(
                                pattern_center_vectors.cbegin(),
                                pattern_center_vectors.cend(),
                                Vector3d::Zero().eval()
                            ) / pattern_center_vectors.size();
        
        Vector3d mean_evec_Z =
            std::accumulate(pattern_evecs_Z.cbegin(), pattern_evecs_Z.cend(), Vector3d::Zero().eval())
            / pattern_evecs_Z.size();
        Vector3d mean_evec_Y =
            std::accumulate(pattern_evecs_Y.cbegin(), pattern_evecs_Y.cend(), Vector3d::Zero().eval())
            / pattern_evecs_Y.size();        
        Vector3d mean_evec_X =
            std::accumulate(pattern_evecs_X.cbegin(), pattern_evecs_X.cend(), Vector3d::Zero().eval())
            / pattern_evecs_X.size();

        print("mean: {}\n", (mean_pcv / 2).eval());
        print("evecsZ: {}\n", mean_evec_Z);
        print("evecsY: {}\n", mean_evec_Y);
        print("evecsX: {}\n", mean_evec_X);

        someplane = Plane((-mean_evec_Z).eval(), (mean_pcv / 2).eval());  // flat plane only one vector in Z direction
        somesinplane = SinusoidalWaveSurface<double>(
            (mean_pcv / 2).eval(),
            mean_evec_X.normalized(),
            mean_evec_Y.normalized(),
            mean_evec_Z.normalized(),
            0, 0, 0 // amplitude, frequency, phase
        );
    
    // Does not work for sin 
    } else if (args.count("abcd")) {
        someplane = Plane(args["abcd"].as<std::vector<double>>());
    
    } else if (args.count("perpvec") && args.count("point")) {
        someplane = Plane(
            args["perpvec"].as<std::vector<double>>(),
            args["point"].as<std::vector<double>>()
        );
    } else {
        throw std::invalid_argument("bad args");
    }

    print("fist guess: {}\n", someplane.abcd());
    print("fist sin guess: {}\n", somesinplane.wave_param());

    // //NOTE: HERE -----------------------------------
    // Vector3<double> intersections;
    // SinusoidalWaveSurface<double> wave_surface = {
    //     {0.1522135796926206, 0.07386223835536908, 0.2721447693021134}, // origin
    //     {-0.8779893174123757, 0.11769968478070637, 0.46398442076461244}, // u    -> propagation is in this direction
    //     {-0.007492214882922366, 0.9658229358896685, -0.25909442916745534}, // v
    //     {0.47911459271736, 0.23195390399035876, 0.8465498174761541}, // normal
    //     0, 0, 0 // amplitude, frequency, phase
    // };
    // Line<double> line({0, 0, 1}, {0,1,1}); // vec , origin
    // // print("sin wave: {}\n", wave_surface.wave_param());
    // bool res = intersectWithSinPlane(line, wave_surface, intersections);
    // print("intersection: {}\n", intersections);
    // //NOTE: HERE -----------------------------------

    std::vector<Plane<double>> steps;
    std::vector<SinusoidalWaveSurface<double>> steps_sin;

    // solve for every combo in multiple steps -> optimization
    if (solve) {
        google::InitGoogleLogging(argv[0]);

        ceres::Problem problem;
        ceres::Solver::Options solver_opts;

        auto abcd_vec = someplane.abcd();
        double abcd[] = {abcd_vec.coeff(0),
                        abcd_vec.coeff(1),
                        abcd_vec.coeff(2),
                        abcd_vec.coeff(3) };

        auto sine_vec = somesinplane.wave_param();
        double sin_params[] = { sine_vec.coeff(0),
                        sine_vec.coeff(1),
                        sine_vec.coeff(2),
                        sine_vec.coeff(3),
                        sine_vec.coeff(4),
                        sine_vec.coeff(5),
                        sine_vec.coeff(6),
                        sine_vec.coeff(7),
                        sine_vec.coeff(8),
                        sine_vec.coeff(9),
                        sine_vec.coeff(10),
                        sine_vec.coeff(11),};

        auto callback = std::make_unique<Recorder>(abcd);
        auto callback_sin = std::make_unique<RecorderSin>(sin_params);
        
        if(!sin){

            auto backrefraction_loss = new ceres::SoftLOneLoss(args["lone"].as<double>());
            auto distance_loss = nullptr; // new ceres::HuberLoss(args["huber"].as<double>());
            
            problem.AddParameterBlock(abcd, 4);
            // Set bounds
            problem.SetParameterLowerBound(abcd, 0, -1.0);
            problem.SetParameterUpperBound(abcd, 0, 1.0);
            problem.SetParameterLowerBound(abcd, 1, -1.0);
            problem.SetParameterUpperBound(abcd, 1, 1.0);
            problem.SetParameterLowerBound(abcd, 2, -1.0);
            problem.SetParameterUpperBound(abcd, 2, 1.0);
            problem.SetParameterLowerBound(abcd, 3, 0.0);
            problem.SetParameterUpperBound(abcd, 3, 40.0);
            
            // iterate through all the combinations of cameras
            for (const auto& [key, combo]: combos) {
                const auto& [T0, T1] = camera_positions[key];

                // iterate through all the triangulated points
                for (const auto& point: triangulations[key]) {

                    // estimates the ditstance between the right and left estimated points
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<EstimatedDistanceCostFunctor, 3, 4>(
                            new EstimatedDistanceCostFunctor { point, T0, T1 },
                            ceres::Ownership::TAKE_OWNERSHIP
                        ),
                        distance_loss,
                        abcd
                    );
                    
                    // calculates the distance to the left camera of the backrefraction
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<BackrefractionCostFunctor<true>, 3, 4>(
                            new BackrefractionCostFunctor<true> {point, T0, T1 },
                            ceres::Ownership::TAKE_OWNERSHIP
                        ),
                        backrefraction_loss,
                        abcd
                    );
                    
                    // calculates the distance to the right camera of the backrefraction
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<BackrefractionCostFunctor<false>, 3, 4>(
                            new BackrefractionCostFunctor<false>{ point, T0, T1 },
                            ceres::Ownership::TAKE_OWNERSHIP
                        ),
                        backrefraction_loss,
                        abcd
                    );
                }
            }

            solver_opts.num_threads = sysconf(_SC_NPROCESSORS_ONLN);
            solver_opts.minimizer_progress_to_stdout = true;
            solver_opts.update_state_every_iteration = true;
            solver_opts.callbacks = { callback.get() };
    
            solver_opts.max_num_iterations = INT_MAX;
            solver_opts.function_tolerance = 1e-30;
            solver_opts.parameter_tolerance = 1e-30;
            solver_opts.gradient_tolerance = 1e-30;

    
            solver_opts.minimizer_type = ceres::MinimizerType::TRUST_REGION;
            solver_opts.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
            solver_opts.use_explicit_schur_complement = true; 

            steps = callback->consume();

        }

        // if sin wave 
        else
        {

            auto backrefraction_loss = new ceres::SoftLOneLoss(args["lone"].as<double>());
            auto distance_loss = nullptr; // new ceres::HuberLoss(args["huber"].as<double>());
            
            problem.AddParameterBlock(sin_params, 12);

            // set bound
            // origin
            problem.SetParameterLowerBound(sin_params, 0, -1.0);
            problem.SetParameterUpperBound(sin_params, 0, 1.0);
            problem.SetParameterLowerBound(sin_params, 1, -1.0);
            problem.SetParameterUpperBound(sin_params, 1, 1.0);
            problem.SetParameterLowerBound(sin_params, 2, 0.0);
            problem.SetParameterUpperBound(sin_params, 2, 1.0);

            // propagation direction
            problem.SetParameterLowerBound(sin_params, 3, -1.0);
            problem.SetParameterUpperBound(sin_params, 3, 1.0);
            problem.SetParameterLowerBound(sin_params, 4, -1.0);
            problem.SetParameterUpperBound(sin_params, 4, 1.0);  
            problem.SetParameterLowerBound(sin_params, 5, -1.0);
            problem.SetParameterUpperBound(sin_params, 5, 1.0);

            // normal -> Note: need to ensure that it is pointing towards the cameras
            problem.SetParameterLowerBound(sin_params, 6, 0.3);
            problem.SetParameterUpperBound(sin_params, 6, 0.6);
            problem.SetParameterLowerBound(sin_params, 7, 0.1);
            problem.SetParameterUpperBound(sin_params, 7, 0.3);  
            problem.SetParameterLowerBound(sin_params, 8, 0.7);
            problem.SetParameterUpperBound(sin_params, 8, 1.0);

            // amplitude
            problem.SetParameterLowerBound(sin_params, 9, 0.0);
            problem.SetParameterUpperBound(sin_params, 9, 0.1);

            // frequency
            problem.SetParameterLowerBound(sin_params, 10, 0.0);
            problem.SetParameterUpperBound(sin_params, 10, 0.1);  

            // phase
            problem.SetParameterLowerBound(sin_params, 11, 0.0);
            problem.SetParameterUpperBound(sin_params, 11, 0.1);
            
            // iterate through all the combinations of cameras
            for (const auto& [key, combo]: combos) {
                const auto& [T0, T1] = camera_positions[key];

                // iterate through all the triangulated points
                for (const auto& point: triangulations[key]) {

                    // estimates the ditstance between the right and left estimated points
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<EstimatedDistanceCostFunctorSinWave, 3, 12>(
                            new EstimatedDistanceCostFunctorSinWave{ point, T0, T1 },
                            ceres::Ownership::TAKE_OWNERSHIP
                        ),
                        distance_loss,
                        sin_params
                    );
                    
                    // calculates the distance to the left camera of the backrefraction
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<BackrefractionCostFunctorSin<true>, 3, 12>(
                            new BackrefractionCostFunctorSin<true> {point, T0, T1 },
                            ceres::Ownership::TAKE_OWNERSHIP
                        ),
                        backrefraction_loss,
                        sin_params
                    );
                    
                    // calculates the distance to the right camera of the backrefraction
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<BackrefractionCostFunctorSin<false>, 3, 12>(
                            new BackrefractionCostFunctorSin<false>{ point, T0, T1 },
                            ceres::Ownership::TAKE_OWNERSHIP
                        ),
                        backrefraction_loss,
                        sin_params
                    );
                }
            }

            solver_opts.num_threads = sysconf(_SC_NPROCESSORS_ONLN);
            solver_opts.minimizer_progress_to_stdout = true;
            solver_opts.update_state_every_iteration = true;
            solver_opts.callbacks = { callback_sin.get() };
    
            solver_opts.max_num_iterations = INT_MAX;
            solver_opts.function_tolerance = 1e-30;
            solver_opts.parameter_tolerance = 1e-30;
            solver_opts.gradient_tolerance = 1e-30;
    
            solver_opts.minimizer_type = ceres::MinimizerType::TRUST_REGION;
            solver_opts.line_search_direction_type = ceres::LineSearchDirectionType::LBFGS;
            solver_opts.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
            solver_opts.use_explicit_schur_complement = true; 

            steps_sin = callback_sin->consume();
        }
            
        // ceres::Solver::Options solver_opts;
        // solver_opts.num_threads = sysconf(_SC_NPROCESSORS_ONLN);
        // solver_opts.minimizer_progress_to_stdout = true;
        // solver_opts.update_state_every_iteration = true;
        // solver_opts.callbacks = { callback.get() };

        // solver_opts.max_num_iterations = INT_MAX;
        // solver_opts.function_tolerance = 1e-30;
        // solver_opts.parameter_tolerance = 1e-20;

        // solver_opts.minimizer_type = ceres::MinimizerType::TRUST_REGION;
        // solver_opts.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
        // solver_opts.use_explicit_schur_complement = true; 

        // solver_opts.line_search_direction_type = ceres::LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT;

        std::string error;

        if (!solver_opts.IsValid(&error)) {
            print(stderr, "error : {}\n", error);
            return 1;
        }

        ceres::Jet<double, 5> j;

        ceres::Solver::Summary summary;

        ceres::Solve(solver_opts, &problem, &summary);
        
        print("ceres report: {}\n", summary.FullReport());

        someplane = {abcd};
        print("best plane: {}\n", someplane.abcd());
        somesinplane = {sin_params};
        print("best sin plane: {}\n", somesinplane.wave_param());

        steps = callback->consume();
        steps_sin = callback_sin->consume();
    }

    // print the results to the serial as json
    if(!sin){

        // create json file for passing to the python program
        steps.push_back(someplane);

        json serialized = { { "steps", json::array() } };

        for (size_t i = 0; i < steps.size(); ++i) {
            const auto& plane = steps[i];

            serialized["steps"].push_back({ { "plane", plane }, { "stereopairs", json::array() } });
            auto& stereopairs = serialized["steps"][i]["stereopairs"];

            for (const auto& [_, combo]: combos) {
                Vectors3d lestimates, restimates, lisects, risects, rbacks, lbacks, rbackrefrs,
                    lbackrefrs;

                const auto warped3D = combo.triangulate_into_referece_frame(detection_type);
                const auto [T0, T1] = combo.get_baseline_in_reference_frame();

                for (const auto& point: warped3D) {
                    Vector3d lestimate, restimate, lisect, risect, rback, lback, rbackrefr, lbackrefr;

                    forward_refract_estimate<double>(point, T0, T1, plane, lestimate, restimate, lisect, risect);
                    back_refract<double>(lestimate, risect, T1, plane, rback, &rbackrefr);
                    back_refract<double>(restimate, lisect, T0, plane, lback, &lbackrefr);

                    lestimates.push_back(lestimate);
                    restimates.push_back(restimate);
                    lisects.push_back(lisect);
                    risects.push_back(risect);
                    rbacks.push_back(rback);
                    lbacks.push_back(lback);
                    rbackrefrs.push_back(rbackrefr);
                    lbackrefrs.push_back(lbackrefr);
                }

                stereopairs.push_back(json { { "idxs", combo.idxs },
                                            { "scenepoints", warped3D },
                                            { "lestimates", lestimates },
                                            { "restimates", restimates },
                                            { "T0", T0 },
                                            { "T1", T1 },
                                            { "lback", lbacks },
                                            { "rback", rbacks },
                                            { "lbackrefr", lbackrefrs },
                                            { "rbackrefr", rbackrefrs },
                                            { "lisects", lisects },
                                            { "risects", risects } });
            }
        }

        print("DELIMITER{}\n", serialized.dump());

    }
    else{
        
        // create json file for passing to the python program
        steps_sin.push_back(somesinplane);

        json serialized = { { "steps", json::array() } };

        for (size_t i = 0; i < steps_sin.size(); ++i) {
            const auto& plane = steps_sin[i];

            serialized["steps"].push_back({ { "plane", plane.wave_param() }, { "stereopairs", json::array() } });
            auto& stereopairs = serialized["steps"][i]["stereopairs"];

            for (const auto& [_, combo]: combos) {
                
                Vectors3d lestimates, restimates, lisects, risects, rbacks, lbacks, rbackrefrs,
                    lbackrefrs;

                const auto warped3D = combo.triangulate_into_referece_frame(detection_type);
                const auto [T0, T1] = combo.get_baseline_in_reference_frame();

                for (const auto& point: warped3D) {
                    
                    Vector3d lestimate, restimate, lisect, risect, rback, lback, rbackrefr, lbackrefr;

                    forward_refract_estimate_sin<double>(point, T0, T1, plane, lestimate, restimate, lisect, risect);
                    back_refract_sin<double>(lestimate, risect, T1, plane, rback, &rbackrefr);
                    back_refract_sin<double>(restimate, lisect, T0, plane, lback, &lbackrefr);

                    // print("lestimate: {}\n", lestimate);
                    // print("restimate: {}\n", restimate);
                    // print("lback: {}\n", lback);
                    // print("rback: {}\n", rback);

                    lestimates.push_back(lestimate);
                    restimates.push_back(restimate);
                    lisects.push_back(lisect);
                    risects.push_back(risect);
                    rbacks.push_back(rback);
                    lbacks.push_back(lback);
                    rbackrefrs.push_back(rbackrefr);
                    lbackrefrs.push_back(lbackrefr);
                }

                stereopairs.push_back(json { { "idxs", combo.idxs },
                                            { "scenepoints", warped3D },
                                            { "lestimates", lestimates },
                                            { "restimates", restimates },
                                            { "T0", T0 },
                                            { "T1", T1 },
                                            { "lback", lbacks },
                                            { "rback", rbacks },
                                            { "lbackrefr", lbackrefrs },
                                            { "rbackrefr", rbackrefrs },
                                            { "lisects", lisects },
                                            { "risects", risects } });
            }
        }

        print("DELIMITER{}\n", serialized.dump());
    }
}
