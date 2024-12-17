#include <iostream>
#include <algorithm>
#include <print>
#include <set>
#include <stdexcept>

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <boost/range/algorithm/transform.hpp>
#include <boost/range/combine.hpp>
#include <nlohmann/json.hpp>

#include "cctag/Detection.hpp"

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

    std::ranges::transform(l, std::back_inserter(lids), idmap);
    std::ranges::transform(r, std::back_inserter(rids), idmap);

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
    Eigen::Vector3d dir, pt;
    Line(const Eigen::Vector3d &direction, const Eigen::Vector3d &point) : dir(direction.normalized()), pt(point) { }
    auto distance_to(const Eigen::Vector3d &other_pt) const {
        double proj_len = (other_pt - pt).dot(dir);
        auto closest_pt = pt + proj_len * dir;
        return other_pt - closest_pt;
    }
    Eigen::Matrix4d pluecker() {
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
    Eigen::Vector3d pt;
    Eigen::Vector4d abcd;
    Plane(const Eigen::Vector3d &perpvec, const Eigen::Vector3d &point) : pt(point) {
        auto norm = perpvec.normalized();
        double d = point.dot(-norm);
        abcd << norm, d;
    }

    Eigen::Vector4d intersect_with_(Line &line) const {
        return line.pluecker().transpose() * abcd;
    }

    Eigen::Vector3d intersect_with(Line &line) const {
        auto hom = intersect_with_(line);
        if (almost_zero(hom.w())) 
            return Eigen::Vector3d { infinity, infinity, infinity };
        return hom.head(3) / hom.w();
    }

    Eigen::Vector3d refract(const Line &line, bool backwards = false) const {
        double r = 1.333; // air -> water
        Eigen::Vector3d n = abcd.head(3);
        if (backwards) {
            r = 1 / r;
            n = -n;
        }

        double c = -n.dot(line.dir);
        return r * line.dir + n * (r * c - std::sqrt(1 - r * r * (1 - c * c)));
    }
};

void to_json(nlohmann::json &j, const Plane &p) {
    j = {
        {"pt", p.pt},
        {"abcd", p.abcd}
    };
}

void forward_refract_estimate(
    const std::vector<Eigen::Vector3d> &scene_pts,
    const Eigen::Vector3d &baseline,
    const Plane &plane,
    std::vector<Eigen::Vector3d> &lestimates,
    std::vector<Eigen::Vector3d> &restimates,
    std::vector<Eigen::Vector3d> &lisects,
    std::vector<Eigen::Vector3d> &risects
) {
    std::vector<Line> llines, rlines;
    std::vector<Eigen::Vector3d> lrefr, rrefr;

    llines.reserve(scene_pts.size());
    rlines.reserve(scene_pts.size());

    std::ranges::transform(scene_pts, std::back_inserter(llines), [](const auto &pt) {
        return Line(pt, Eigen::Vector3d::Zero());
    });
    std::ranges::transform(scene_pts, std::back_inserter(rlines), [&baseline](const auto &pt) {
        return Line(pt - baseline, baseline);
    });

    lisects.clear();
    risects.clear();

    std::ranges::transform(llines, std::back_inserter(lisects), [&plane](auto &l) {
        return plane.intersect_with(l);
    });
    std::ranges::transform(rlines, std::back_inserter(risects), [&plane](auto &l) {
        return plane.intersect_with(l);
    });

    std::ranges::transform(llines, std::back_inserter(lrefr), [&plane](const auto &l) {
        return plane.refract(l);
    });
    std::ranges::transform(rlines, std::back_inserter(rrefr), [&plane](const auto &l) {
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


int main(int argc, const char **argv) {
    if (argc != 6) {
        std::println(std::cerr, "Usage: {} [limg] [rimg] [lcal] [rcal] [april/cctag]", argv[0]);
        return 1;
    }

    std::string tag = argv[5];

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
    std::vector<Eigen::Vector3d> warped3D;
    Eigen::Vector3d T { T_.val };
    T = -T;

    cv::triangulatePoints(lP, rP, ldects, rdects, warped3D_);

    for (int col = 0; col < warped3D_.cols; ++col) {
        cv::Vec4f hom_pt = warped3D_.col(col);
        auto [x, y, z, w] = hom_pt.val;
        warped3D.push_back(Eigen::Vector3d {
            x/w, y/w, z/w
        });
    }

    std::vector<Eigen::Vector3d> lestimates, restimates, lisects, risects;

    Plane someplane(
        Eigen::Vector3d { 0.0, -0.25, -0.5 },
        Eigen::Vector3d { 0.2, -0.05, 0.4 }
    );

    forward_refract_estimate(warped3D, T, someplane, lestimates, restimates, lisects, risects);

    nlohmann::json serialized = {
        {"scenepoints", warped3D  },
        {"someplane",   someplane },
        {"lestimates",  lestimates},
        {"restimates",  restimates},
        {"baseline",    T         }
    };

    std::println("{}", serialized.dump());
}
