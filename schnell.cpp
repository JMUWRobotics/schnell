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
using std::println;
using std::format;
#else
template<typename Type, int Size>
struct fmt::formatter<Eigen::Vector<Type, Size>>: fmt::formatter<std::string> {
    auto format(const Eigen::Vector<Type, Size>& v, fmt::format_context& ctx) const {
        return fmt::formatter<std::string>::format(
            std::accumulate(
                std::next(v.begin()),
                v.end(),
                fmt::format("[{}", v[0]),
                [](std::string a, const Type& x) { return fmt::format("{}, {}", std::move(a), x); }
            ) + ']',
            ctx
        );
    }
};
using fmt::println;
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

template<typename T>
class Line {
    std::optional<Eigen::Matrix4<T>> _pluecker;

public:
    Vector3<T> dir, pt;
    Line(const Vector3<T>& direction, const Vector3<T>& point):
        dir(direction.normalized()),
        pt(point) {}
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
};

template<typename T>
struct Plane {
    Vector3<T> abc;
    T d;

    Plane(): abc(Vector3<T>::Zero()) {}

    Plane(const Vector3<T>& perpvec, const Vector3<T>& point) {
        // a(x - px) + b(y - py) + c(z - pz) = 0
        // d = -a*px - b*py - c*pz
        // ax + by + cz + (-a*px -b*py -c*pz) = 0
        abc = perpvec.normalized();
        d = point.dot(-abc);
    }

    Plane(const std::vector<T>& perpvec, const std::vector<T>& point):
        Plane(
            Vector3<T> { perpvec.at(0), perpvec.at(1), perpvec.at(2) },
            Vector3<T> { point.at(0), point.at(1), point.at(2) }
        ) {}

    Plane(T a, T b, T c, T d): abc(a, b, c), d(d) {
        abc.normalize();
    }

    Plane(const T* abcd): Plane(abcd[0], abcd[1], abcd[2], abcd[3]) {}

    Plane(const std::vector<T>& abcd): Plane(abcd.at(0), abcd.at(1), abcd.at(2), abcd.at(3)) {}

    Vector4<T> abcd() const {
        return { abc.x(), abc.y(), abc.z(), d };
    }

    Vector3<T> some_point() const {
        Vector3<T> ret;
        size_t maxidx;

        abc.head(3).cwiseAbs().maxCoeff(&maxidx);

        double max = abc[maxidx];

        if (almost_zero(max))
            println(stderr, "Almost zero maximum coefficient");

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
    Vector3<T>& risect
) {
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

struct MyCostFunctor {
    const Vector3d T0, T1;
    const Vector3d scene;

    MyCostFunctor(const Vector3d& warped, const Vector3d& T0, const Vector3d& T1):
        T0(T0),
        T1(T1),
        scene(warped) {}
};

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

template<typename TVal>
using StereoMap = std::map<std::tuple<int, int>, TVal>;

int main(int argc, const char** argv) {
    // clang-format off

    cxxopts::Options options("schnell");
    options.add_options()
        ("datapath", "path to data", cxxopts::value<std::string>())
        ("abcd", "initial plane values", cxxopts::value<std::vector<double>>())
        ("point", "xyz point on plane", cxxopts::value<std::vector<double>>())
        ("perpvec", "xyz components of vector perpendicular to plane", cxxopts::value<std::vector<double>>())
        ("sift", "enable sift detection")
        ("solve", "runs solver")
        //("huber", "huber loss coefficient", cxxopts::value<double>()->default_value("0.1"))
        ("lone", "softlone loss coefficient", cxxopts::value<double>()->default_value("0.1"));

    options.parse_positional({"datapath"});

    const auto args = options.parse(argc, argv);

    // clang-format on

    bool solve = args.count("solve");
    DetectionType detection_type = args.count("sift") ? DetectionType::SIFT : DetectionType::APRIL;

    std::string datapath = args["datapath"].as<std::string>();

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

    bool guess_plane = !(args.count("abcd") || (args.count("perpvec") && args.count("point")));

    Plane<double> someplane;

    StereoMap<Vectors3d> triangulations;
    StereoMap<std::tuple<Vector3d, Vector3d>> camera_positions;
    for (const auto& [key, combo]: combos) {
        camera_positions[key] = combo.get_baseline_in_reference_frame();
        triangulations[key] = combo.triangulate_into_referece_frame(detection_type);
    }

    if (guess_plane) {
        Vectors3d pattern_center_vectors;
        Vectors3d pattern_evecs;

        for (const auto& [key, combo]: combos) {
            const auto& [T0, T1] = camera_positions[key];
            const auto [mean, evecs] = principal_components(triangulations[key]);
            pattern_center_vectors.push_back(mean - T0);
            pattern_center_vectors.push_back(mean - T1);
            pattern_evecs.push_back(evecs[0]);
        }

        Vector3d mean_pcv = std::accumulate(
                                pattern_center_vectors.cbegin(),
                                pattern_center_vectors.cend(),
                                Vector3d::Zero().eval()
                            )
            / pattern_center_vectors.size();
        Vector3d mean_evec =
            std::accumulate(pattern_evecs.cbegin(), pattern_evecs.cend(), Vector3d::Zero().eval())
            / pattern_evecs.size();

        someplane = Plane((-mean_evec).eval(), (mean_pcv / 2).eval());
    } else if (args.count("abcd")) {
        someplane = Plane(args["abcd"].as<std::vector<double>>());
    } else if (args.count("perpvec") && args.count("point")) {
        someplane = Plane(
            args["perpvec"].as<std::vector<double>>(),
            args["point"].as<std::vector<double>>()
        );
    } else
        throw std::invalid_argument("bad args");

    println("{}", someplane.abcd());

    std::vector<Plane<double>> steps;

    if (solve) {
        google::InitGoogleLogging(argv[0]);

        ceres::Problem problem;

        auto abcd_vec = someplane.abcd();
        double abcd[] = { abcd_vec.coeff(0),
                          abcd_vec.coeff(1),
                          abcd_vec.coeff(2),
                          abcd_vec.coeff(3) };

        auto backrefraction_loss = new ceres::SoftLOneLoss(args["lone"].as<double>());
        auto distance_loss = nullptr; // new ceres::HuberLoss(args["huber"].as<double>());

        for (const auto& [key, combo]: combos) {
            const auto& [T0, T1] = camera_positions[key];

            for (const auto& point: triangulations[key]) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<EstimatedDistanceCostFunctor, 3, 4>(
                        new EstimatedDistanceCostFunctor { point, T0, T1 },
                        ceres::Ownership::TAKE_OWNERSHIP
                    ),
                    distance_loss,
                    abcd
                );

                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<BackrefractionCostFunctor<true>, 3, 4>(
                        new BackrefractionCostFunctor<true> { point, T0, T1 },
                        ceres::Ownership::TAKE_OWNERSHIP
                    ),
                    backrefraction_loss,
                    abcd
                );

                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<BackrefractionCostFunctor<false>, 3, 4>(
                        new BackrefractionCostFunctor<false> { point, T0, T1 },
                        ceres::Ownership::TAKE_OWNERSHIP
                    ),
                    backrefraction_loss,
                    abcd
                );
            }
        }

        auto callback = std::make_unique<Recorder>(abcd);

        ceres::Solver::Options solver_opts;
        solver_opts.num_threads = sysconf(_SC_NPROCESSORS_ONLN);
        solver_opts.minimizer_progress_to_stdout = true;
        solver_opts.update_state_every_iteration = true;
        solver_opts.callbacks = { callback.get() };

        solver_opts.max_num_iterations = INT_MAX;
        solver_opts.function_tolerance = 1e-30;
        solver_opts.parameter_tolerance = 1e-20;

        // solver_opts.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
        // solver_opts.line_search_direction_type =
        //     ceres::LineSearchDirectionType::NONLINEAR_CONJUGATE_GRADIENT;

        solver_opts.minimizer_type = ceres::MinimizerType::TRUST_REGION;
        //solver_opts.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;

        std::string error;
        if (!solver_opts.IsValid(&error)) {
            println(stderr, "{}", error);
            return 1;
        }

        ceres::Jet<double, 5> j;

        ceres::Solver::Summary summary;

        ceres::Solve(solver_opts, &problem, &summary);

        println("{}", summary.FullReport());

        someplane = { abcd };
        steps = callback->consume();
    }

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

                forward_refract_estimate<
                    double>(point, T0, T1, plane, lestimate, restimate, lisect, risect);
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

    println("DELIMITER{}", serialized.dump());
}
