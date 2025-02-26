#include <algorithm>
#include <ceres/jet_fwd.h>
#include <iostream>
#include <print>
#include <set>
#include <stdexcept>

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/types.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <boost/range/algorithm/transform.hpp>
#include <boost/range/combine.hpp>

#include <nlohmann/json.hpp>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#define BOOST_ALLOW_DEPRECATED_HEADERS

#include "cctag/Detection.hpp" // noisy fucking library

#undef BOOST_ALLOW_DEPRECATED_HEADERS
#pragma GCC diagnostic pop

using Eigen::Vector3;
using Eigen::Vector3d;
using Eigen::Vector4;
using Vectors3d = std::vector<Vector3d>;
using std::ranges::transform;

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
                    out.push_back({ d[i].p[j][0], d[i].p[j][1] });
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

    zarray_t *lz = apriltag_detector_detect(d, &lapr), *rz = apriltag_detector_detect(d, &rapr);

    std::vector<apriltag_detection_t> ld = dectvec_from_zarray(std::move(lz)),
                                      rd = dectvec_from_zarray(std::move(rz));

    std::set<int> ids;

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

auto read_PT(const char* path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);

    cv::Matx34d P;
    cv::Matx31d T;
    fs["rect_proj"] >> P;
    fs["T"] >> T;

    fs.release();

    return std::make_tuple(P, T);
}

template<typename T>
constexpr T infinity = T(std::numeric_limits<double>::infinity());

constexpr bool almost_zero(double x) {
    // https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#
    constexpr double atol = 1e-08;
    return std::abs(x) <= atol;
}

template<typename T>
class Line {
    std::optional<Eigen::Matrix4<T>> _pluecker;

public:
    Vector3<T> dir, pt;
    Line(const Vector3<T>& direction, const Vector3<T>& point):
        dir(direction.normalized()),
        pt(point) {}
    auto distance_to(const Vector3<T>& other_pt) const {
        auto proj_len = (other_pt - pt).dot(dir);
        auto closest_pt = pt + proj_len * dir;
        return other_pt - closest_pt;
    }
    const auto& pluecker() {
        if (!_pluecker.has_value()) {
            Vector4<T> a = (pt + dir).homogeneous(), b = pt.homogeneous();

            _pluecker = a * b.transpose() - b * a.transpose();
        }

        return _pluecker.value();
    }
};

template<typename T>
struct Plane {
    Eigen::Vector4<T> abcd;
    Plane(const Vector3<T>& perpvec, const Vector3<T>& point) {
        // a(x - px) + b(y - py) + c(z - pz) = 0
        // d = -a*px - b*py - c*pz
        auto norm = perpvec.normalized();
        T d = point.dot(-norm);
        abcd << norm, d;
    }

    Plane(const T* abcd): abcd(abcd[0], abcd[1], abcd[2], abcd[3]) {}

    Vector3<T> some_point() const {
        Vector3<T> ret;
        size_t maxidx;

        abcd.head(3).cwiseAbs().maxCoeff(&maxidx);

        double max = abcd[maxidx];

        if (almost_zero(max))
            std::println(stderr, "Almost zero maximum coefficient");

        double d = abcd.tail(1).value();
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

    Eigen::Vector4<T> intersect_with_(Line<T>& line) const {
        return line.pluecker().transpose() * abcd;
    }

    Vector3<T> intersect_with(Line<T>& line) const {
        auto hom = intersect_with_(line);
        if constexpr (std::is_floating_point_v<T>)
            if (almost_zero(hom.w()))
                return Vector3<T> { infinity<T>, infinity<T>, infinity<T> };
        return hom.hnormalized();
    }

    Vector3<T> refract(const Line<T>& line, bool backwards = false) const {
        double r = 1.33; // air -> water
        Vector3<T> n = abcd.head(3);

        // TODO make sense of this

        if (backwards) {
            n.x() *= -1.0;
            n.y() *= -1.0;
            n.z() *= -1.0;
        } else
            r = 1.0 / r;

        T c = -n.dot(line.dir);
        return r * line.dir + n * (r * c - ceres::sqrt(1.0 - r * r * (1.0 - c * c)));
    }
};

void to_json(nlohmann::json& j, const Plane<double>& p) {
    j = { { "pt", p.some_point() }, { "abcd", p.abcd } };
}

template<typename T>
void forward_refract_estimate(
    const Vector3<T>& pt,
    const Vector3<T>& baseline,
    const Plane<T>& plane,
    Vector3<T>& lestimate,
    Vector3<T>& restimate,
    Vector3<T>& lisect,
    Vector3<T>& risect
) {
    Line<T> lline(pt, Vector3<T>::Zero()), rline(pt - baseline, baseline);

    lisect = plane.intersect_with(lline);
    risect = plane.intersect_with(rline);

    auto lrefr = plane.refract(lline), rrefr = plane.refract(rline);

    Plane rrefrplane(rline.dir.cross(rrefr), risect), lrefrplane(lline.dir.cross(lrefr), lisect);

    Line lrefrline(lrefr, lisect), rrefrline(rrefr, risect);

    lestimate = rrefrplane.intersect_with(lrefrline);
    restimate = lrefrplane.intersect_with(rrefrline);
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
    const Vector3d& baseline;
    const Vector3d& scene;

    MyCostFunctor(const Vector3d& warped, const Vector3d& baseline):
        baseline(baseline),
        scene(warped) {}
};

struct EstimatedDistanceCostFunctor: public MyCostFunctor {
    using MyCostFunctor::MyCostFunctor;

    template<typename T>
    bool operator()(const T* const abcd, T* residuals) const {
        const Plane someplane(abcd);

        Vector3<T> _lestimates, _restimates, _lisects, _risects, _back;

        forward_refract_estimate<T>(
            scene.cast<T>(),
            baseline.cast<T>(),
            someplane,
            _lestimates,
            _restimates,
            _lisects,
            _risects
        );

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

        forward_refract_estimate<T>(
            scene.cast<T>(),
            baseline.cast<T>(),
            someplane,
            _lestimates,
            _restimates,
            _lisects,
            _risects
        );

        if constexpr (LEFT)
            back_refract<T>(_lestimates, _risects, baseline.cast<T>(), someplane, _back, nullptr);
        else
            back_refract<T>(
                _restimates,
                _lisects,
                Vector3<T>::Zero(),
                someplane,
                _back,
                nullptr
            );

        residuals[0] = _back.x();
        residuals[1] = _back.y();
        residuals[2] = _back.z();

        return true;
    }
};

int main(int argc, const char** argv) {
    if (argc != 7) {
        std::println(
            std::cerr,
            "Usage: {} [limg] [rimg] [lcal] [rcal] [april/cctag] [demo/solve]",
            argv[0]
        );
        return 1;
    }

    std::string tag = argv[5], mode = argv[6];

    cv::Mat_<uint8_t> limg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE),
                      rimg = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    std::vector<cv::Vec2d> ldects, rdects;
    if (tag == "cctag")
        detect_cctags(limg, rimg, ldects, rdects);
    else
        detect_apriltags(limg, rimg, ldects, rdects);

    if (ldects.empty() || rdects.empty())
        throw std::invalid_argument("No detections");

    auto [lP, _] = read_PT(argv[3]);
    auto [rP, T_] = read_PT(argv[4]);

    cv::Mat warped3D_;
    Vectors3d warped3D;
    // - Tx*f / f
    Vector3d T { -rP(0, 3) / rP(0, 0), 0.0, 0.0 };

    cv::triangulatePoints(lP, rP, ldects, rdects, warped3D_);

    for (int col = 0; col < warped3D_.cols; ++col) {
        cv::Vec4f hom_pt = warped3D_.col(col);
        auto [x, y, z, w] = hom_pt.val;
        warped3D.push_back(Vector3d { x / w, y / w, z / w });
    }

    Vectors3d lestimates, restimates, lisects, risects, rbacks, lbacks, rbackrefrs, lbackrefrs;

    Plane someplane(Vector3d { 0.0, -0.25, -0.5 }, Vector3d { 0.2, -0.05, 0.4 });

    if (mode == "solve") {
        google::InitGoogleLogging(argv[0]);

        ceres::Problem problem;

        double abcd[] = { someplane.abcd.coeff(0),
                          someplane.abcd.coeff(1),
                          someplane.abcd.coeff(2),
                          someplane.abcd.coeff(3) };

        for (const auto& point: warped3D) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<EstimatedDistanceCostFunctor, 3, 4>(
                    new EstimatedDistanceCostFunctor { point, T },
                    ceres::Ownership::TAKE_OWNERSHIP
                ),
                nullptr,
                abcd
            );

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<BackrefractionCostFunctor<true>, 3, 4>(
                    new BackrefractionCostFunctor<true> { point, T },
                    ceres::Ownership::TAKE_OWNERSHIP
                ),
                nullptr,
                abcd
            );

            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<BackrefractionCostFunctor<false>, 3, 4>(
                    new BackrefractionCostFunctor<false> { point, T },
                    ceres::Ownership::TAKE_OWNERSHIP
                ),
                nullptr,
                abcd
            );
        }

        ceres::Solver::Options solver_opts;
        solver_opts.minimizer_progress_to_stdout = true;

        std::string error;
        if (!solver_opts.IsValid(&error)) {
            std::println(stderr, "{}", error);
            return 1;
        }

        ceres::Jet<double, 5> j;

        ceres::Solver::Summary summary;

        ceres::Solve(solver_opts, &problem, &summary);

        std::println("{}", summary.BriefReport());

        someplane = { abcd };
    }

    for (const auto& point: warped3D) {
        Vector3d lestimate, restimate, lisect, risect, rback, lback, rbackrefr, lbackrefr;

        forward_refract_estimate<double>(point, T, someplane, lestimate, restimate, lisect, risect);
        back_refract<double>(lestimate, risect, T, someplane, rback, &rbackrefr);
        back_refract<double>(restimate, lisect, Vector3d::Zero(), someplane, lback, &lbackrefr);

        lestimates.push_back(lestimate);
        restimates.push_back(restimate);
        lisects.push_back(lisect);
        risects.push_back(risect);
        rbacks.push_back(rback);
        lbacks.push_back(lback);
        rbackrefrs.push_back(rbackrefr);
        lbackrefrs.push_back(lbackrefr);
    }

    nlohmann::json serialized = {
        { "scenepoints", warped3D },  { "someplane", someplane },  { "lestimates", lestimates },
        { "restimates", restimates }, { "baseline", T },           { "lback", lbacks },
        { "rback", rbacks },          { "lbackrefr", lbackrefrs }, { "rbackrefr", rbackrefrs },
        { "lisects", lisects },       { "risects", risects }
    };

    std::println("DELIMITER{}", serialized.dump());
}
