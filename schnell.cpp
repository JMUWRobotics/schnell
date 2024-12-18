#include <ceres/jet_fwd.h>
#include <iostream>
#include <algorithm>
#include <print>
#include <set>
#include <stdexcept>

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/types.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

#include <boost/range/algorithm/transform.hpp>
#include <boost/range/combine.hpp>

#include <nlohmann/json.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#define BOOST_ALLOW_DEPRECATED_HEADERS

#include "cctag/Detection.hpp" // noisy fucking library

#undef BOOST_ALLOW_DEPRECATED_HEADERS
#pragma GCC diagnostic pop

using Eigen::Vector3d;
using Vectors3d = std::vector<Vector3d>;
using std::ranges::transform;

image_u8_t april_from_mat(const cv::Mat_<uint8_t>& m) {
    return image_u8_t {
        .width  = m.cols,
        .height = m.rows,
        .stride = (int)m.step,
        .buf    = m.data
    };
}

std::vector<apriltag_detection_t> dectvec_from_zarray(zarray_t *&&z) {
    std::vector<apriltag_detection_t> ret;
    apriltag_detection_t *d;

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
    std::vector<apriltag_detection_t> &l,
    std::vector<apriltag_detection_t> &r,
    std::vector<cv::Vec2d> &lout,
    std::vector<cv::Vec2d> &rout
) {
    static const auto idcomp = [](const auto& l, const auto& r) { return l.id < r.id; };
    static const auto idmap = [](const auto &d) { return d.id; };

    std::vector<int> lids, rids;
    std::set<int> isect;

    std::ranges::sort(l, idcomp);
    std::ranges::sort(r, idcomp);

    transform(l, std::back_inserter(lids), idmap);
    transform(r, std::back_inserter(rids), idmap);

    std::ranges::set_intersection(lids, rids, std::inserter(isect, isect.begin()));

    auto fill_out = [&](const auto &ids, const auto &d, auto &out) {
        for (size_t i = 0; i < ids.size(); ++i)
            if (isect.contains(ids[i]))
                for (int j = 0; j < 4; ++j)
                    out.push_back({ d[i].p[j][0], d[i].p[j][1] });
    };

    fill_out(lids, l, lout);
    fill_out(rids, r, rout);    
}

void intersect_cctag_dects(
    boost::ptr_list<cctag::CCTag> &l,
    boost::ptr_list<cctag::CCTag> &r,
    std::vector<cv::Vec2d> &lout,
    std::vector<cv::Vec2d> &rout
) {
    static const auto idcomp = [](const auto& l, const auto& r) { return l.id() < r.id(); };
    static const auto idmap = [](const auto &d) { return d.id(); };

    std::vector<int> lids, rids;
    std::set<int> isect;

    l.sort(idcomp);
    r.sort(idcomp);

    boost::transform(l, std::back_inserter(lids), idmap);
    boost::transform(r, std::back_inserter(rids), idmap);

    std::ranges::set_intersection(lids, rids, std::inserter(isect, isect.begin()));

    auto fill_out = [&](const auto &ids, const auto &d, auto &out) {
        auto dit = d.cbegin();
        for (size_t i = 0; i < ids.size(); ++i, ++dit)
            if (isect.contains(ids[i]))
                out.push_back({ dit->x(), dit->y() });
    };

    fill_out(lids, l, lout);
    fill_out(rids, r, rout);
}

void detect_apriltags(
    const cv::Mat_<uint8_t> &limg,
    const cv::Mat_<uint8_t> &rimg,
    std::vector<cv::Vec2d> &ldects,
    std::vector<cv::Vec2d> &rdects
) {
    apriltag_detector_t *d = apriltag_detector_create();
    apriltag_family_t   *f = tag36h11_create();
    apriltag_detector_add_family(d, f);

    image_u8_t
        lapr = april_from_mat(limg),
        rapr = april_from_mat(rimg);

    zarray_t
        *lz = apriltag_detector_detect(d, &lapr),
        *rz = apriltag_detector_detect(d, &rapr);

    std::vector<apriltag_detection_t>
        ld = dectvec_from_zarray(std::move(lz)),
        rd = dectvec_from_zarray(std::move(rz));

    std::set<int> ids;

    tag36h11_destroy(f);
    apriltag_detector_destroy(d);

    intersect_apriltag_dects(ld, rd, ldects, rdects);
}

void detect_cctags(
    const cv::Mat_<uint8_t> &limg,
    const cv::Mat_<uint8_t> &rimg,
    std::vector<cv::Vec2d> &ldects,
    std::vector<cv::Vec2d> &rdects 
) {
    static const cctag::Parameters cctp { 4 };
    static const cctag::CCTagMarkersBank cctb { cctp._nCrowns };

    boost::ptr_list<cctag::CCTag> ld, rd;
    cctag::cctagDetection(ld, 0, 0, limg, cctp, cctb);
    cctag::cctagDetection(rd, 0, 0, rimg, cctp, cctb);

    intersect_cctag_dects(ld, rd, ldects, rdects);
}

auto read_PT(const char *path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);

    cv::Matx34d P;
    cv::Matx31d T;
    fs["rect_proj"] >> P;
    fs["T"] >> T;

    fs.release();

    return std::make_tuple(P, T);
}

constexpr double infinity = std::numeric_limits<double>::infinity();

constexpr bool almost_zero(double x) {
    // https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#
    constexpr double atol = 1e-08;
    return std::abs(x) <= atol;
}

class Line {
    std::optional<Eigen::Matrix4d> _pluecker;
public:
    Vector3d dir, pt;
    Line(const Vector3d &direction, const Vector3d &point) : dir(direction.normalized()), pt(point) { }
    auto distance_to(const Vector3d &other_pt) const {
        auto proj_len = (other_pt - pt).dot(dir);
        auto closest_pt = pt + proj_len * dir;
        return other_pt - closest_pt;
    }
    const Eigen::Matrix4d& pluecker() {
        if (!_pluecker.has_value()) {
            Eigen::Vector4d a, b;

            a << pt + dir, 1.0;
            b << pt      , 1.0;

            _pluecker = a * b.transpose() - b * a.transpose();
        }

        return _pluecker.value();
    }
};

struct Plane {
    Eigen::Vector4d abcd;
    Plane(const Vector3d &perpvec, const Vector3d &point) {
        // a(x - px) + b(y - py) + c(z - pz) = 0
        // d = -a*px - b*py - c*pz
        auto norm = perpvec.normalized();
        double d = point.dot(-norm);
        abcd << norm, d;
    }

    Plane(const double *abcd) : abcd(abcd[0], abcd[1], abcd[2], abcd[3]) { }

    Vector3d some_point() const {
        Vector3d ret;
        size_t maxidx;

        abcd.head(3).cwiseAbs().maxCoeff(&maxidx);

        double max = abcd[maxidx];

        if (almost_zero(max))
            std::println(stderr, "Almost zero maximum coefficient");

        double d = abcd.tail(1).value();
        switch (maxidx) {
        case 0: // a
            ret.y() = ret.z() = 0.0;
            ret.x() = - d / max;
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

    Eigen::Vector4d intersect_with_(Line &line) const {
        return line.pluecker().transpose() * abcd;
    }

    Vector3d intersect_with(Line &line) const {
        auto hom = intersect_with_(line);
        if (almost_zero(hom.w())) 
            return Vector3d { infinity, infinity, infinity };
        return hom.head(3) / hom.w();
    }

    Vector3d refract(const Line &line, bool backwards = false) const {
        double r = 1.333; // air -> water
        Vector3d n = abcd.head(3);

        // TODO make sense of this
        if (backwards)
            n *= -1;
        else
            r = 1 / r;

        double c = -n.dot(line.dir);
        return r * line.dir + n * (r * c - std::sqrt(1 - r * r * (1 - c * c)));
    }
};

void to_json(nlohmann::json &j, const Plane &p) {
    j = {
        {"pt", p.some_point()},
        {"abcd", p.abcd}
    };
}

void forward_refract_estimate(
    const Vectors3d &scene_pts,
    const Vector3d &baseline,
    const Plane &plane,
    Vectors3d &lestimates,
    Vectors3d &restimates,
    Vectors3d &lisects,
    Vectors3d &risects
) {
    std::vector<Line> llines, rlines;
    Vectors3d lrefr, rrefr;

    llines.reserve(scene_pts.size());
    rlines.reserve(scene_pts.size());

    transform(scene_pts, std::back_inserter(llines), [](const auto &pt) {
        return Line(pt, Vector3d::Zero());
    });
    transform(scene_pts, std::back_inserter(rlines), [&baseline](const auto &pt) {
        return Line(pt - baseline, baseline);
    });

    lisects.clear();
    risects.clear();

    transform(llines, std::back_inserter(lisects), [&plane](auto &l) {
        return plane.intersect_with(l);
    });
    transform(rlines, std::back_inserter(risects), [&plane](auto &l) {
        return plane.intersect_with(l);
    });

    transform(llines, std::back_inserter(lrefr), [&plane](const auto &l) {
        return plane.refract(l);
    });
    transform(rlines, std::back_inserter(rrefr), [&plane](const auto &l) {
        return plane.refract(l);
    });

    lestimates.clear();
    restimates.clear();

    for (const auto &[l, r, lpt, rpt, lline, rline] : boost::combine(lrefr, rrefr, lisects, risects, llines, rlines)) {
        Plane rrefrplane(rline.dir.cross(r), rpt),
              lrefrplane(lline.dir.cross(l), lpt);

        Line lrefrline(l, lpt),
             rrefrline(r, rpt);

        lestimates.push_back(
            rrefrplane.intersect_with(lrefrline)
        );
        restimates.push_back(
            lrefrplane.intersect_with(rrefrline)
        );
    }
}

void back_refract(
    const Vectors3d &estimates,
    const Vectors3d &isects,
    const Vector3d &baseline,
    const Plane &someplane,
    Vectors3d &distances,
    Vectors3d *backrefractions
) {
    static thread_local Vectors3d backrefractions_;
    if (!backrefractions)
        backrefractions = &backrefractions_;
    
    backrefractions->clear();
    distances.clear();

    transform(boost::combine(estimates, isects), std::back_inserter(*backrefractions), [&someplane](const auto &tup) {
        auto &[est, isect] = tup;
        auto l = Line { isect - est, est };
        return someplane.refract(l, true);
    });

    transform(boost::combine(*backrefractions, isects), std::back_inserter(distances), [&](const auto &tup) {
        auto &[refr, isect] = tup;
        return Line(refr, isect).distance_to(baseline);
    });
}

struct NumericCostFunctor {
    const Vector3d &baseline;
    const Vectors3d &scene;

    NumericCostFunctor(const Vectors3d &warped, const Vector3d &baseline)
        : baseline(baseline), scene(warped) { }
};

struct EstimatedDistanceCostFunctor : public NumericCostFunctor {
    using NumericCostFunctor::NumericCostFunctor;

    bool operator() (const double *const abcd, double *residuals) const {
        const Plane someplane(abcd);

        Vectors3d _lestimates, _restimates, _lisects, _risects, _back;

        forward_refract_estimate(scene, baseline, someplane, _lestimates, _restimates, _lisects, _risects);

        const size_t N = _lestimates.size();
        size_t idx = 0;

        for (size_t i = 0; i < N; ++i) {
            auto estdist = _lestimates[i] - _restimates[i];
            for (size_t j = 0; j < 3; j++)
                residuals[idx++] = estdist.coeff(j);
        }

        return true;
    }
};

template<bool LEFT>
struct BackrefractionCostFunctor : public NumericCostFunctor {
    using NumericCostFunctor::NumericCostFunctor;

    bool operator() (const double *const abcd, double *residuals) const {
        const Plane someplane(abcd);

        Vectors3d _lestimates, _restimates, _lisects, _risects, _back;

        forward_refract_estimate(scene, baseline, someplane, _lestimates, _restimates, _lisects, _risects);

        if constexpr (LEFT)
            back_refract(_lestimates, _risects, baseline, someplane, _back, nullptr);
        else
            back_refract(_restimates, _lisects, Vector3d::Zero(), someplane, _back, nullptr);

        const size_t N = _lestimates.size();
        size_t idx = 0;

        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < 3; j++)
                residuals[idx++] = _back[i].coeff(j);

        return true;
    }
};


int main(int argc, const char **argv) {
    if (argc != 7) {
        std::println(std::cerr, "Usage: {} [limg] [rimg] [lcal] [rcal] [april/cctag] [demo/solve]", argv[0]);
        return 1;
    }

    std::string tag  = argv[5],
                mode = argv[6];

    cv::Mat_<uint8_t>
        limg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE),
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
    Vector3d T { - rP(0, 3) / rP(0, 0), 0.0, 0.0 };

    cv::triangulatePoints(lP, rP, ldects, rdects, warped3D_);

    for (int col = 0; col < warped3D_.cols; ++col) {
        cv::Vec4f hom_pt = warped3D_.col(col);
        auto [x, y, z, w] = hom_pt.val;
        warped3D.push_back(Vector3d {
            x/w, y/w, z/w
        });
    }

    Vectors3d lestimates, restimates, lisects, risects, rback, lback, rbackrefr, lbackrefr;

    Plane someplane(
        Vector3d { 0.0, -0.25, -0.5 },
        Vector3d { 0.2, -0.05, 0.4 }
    );

    if (mode == "solve") {
        google::InitGoogleLogging(argv[0]);

        ceres::Problem problem;

        double abcd[] = {
            someplane.abcd.coeff(0), someplane.abcd.coeff(1), someplane.abcd.coeff(2), someplane.abcd.coeff(3)
        };

        int n_residuals = warped3D.size() // number of points
                        * 3;              // 3D

        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<
                EstimatedDistanceCostFunctor, ceres::NumericDiffMethodType::CENTRAL, ceres::DYNAMIC, 4
            >(new EstimatedDistanceCostFunctor { warped3D, T } , ceres::Ownership::TAKE_OWNERSHIP, n_residuals),
            nullptr,
            abcd
        );

        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<
                BackrefractionCostFunctor<true>, ceres::NumericDiffMethodType::CENTRAL, ceres::DYNAMIC, 4
            >(new BackrefractionCostFunctor<true> { warped3D, T } , ceres::Ownership::TAKE_OWNERSHIP, n_residuals),
            nullptr,
            abcd
        );

        problem.AddResidualBlock(
            new ceres::NumericDiffCostFunction<
                BackrefractionCostFunctor<false>, ceres::NumericDiffMethodType::CENTRAL, ceres::DYNAMIC, 4
            >(new BackrefractionCostFunctor<false> { warped3D, T } , ceres::Ownership::TAKE_OWNERSHIP, n_residuals),
            nullptr,
            abcd
        );

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

    forward_refract_estimate(warped3D, T, someplane, lestimates, restimates, lisects, risects);

    back_refract(lestimates, risects, T, someplane, rback, &rbackrefr),
    back_refract(restimates, lisects, Vector3d::Zero(), someplane, lback, &lbackrefr);

    nlohmann::json serialized = {
        {"scenepoints", warped3D  },
        {"someplane",   someplane },
        {"lestimates",  lestimates},
        {"restimates",  restimates},
        {"baseline",    T         },
        {"lback",       lback     },
        {"rback",       rback     },
        {"lbackrefr",   lbackrefr },
        {"rbackrefr",   rbackrefr },
        {"lisects",     lisects   },
        {"risects",    risects    }
    };

    std::println("DELIMITER{}", serialized.dump());
}
