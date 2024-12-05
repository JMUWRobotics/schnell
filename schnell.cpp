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

cv::Matx34d read_P(const char *path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);

    cv::Matx34d P;
    fs["Q"] >> P;

    fs.release();

    return P;
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

    for (size_t i = 0; i < ldects.size(); ++i)
        std::println("({}, {}) | ({}, {})", ldects[i][0], ldects[i][1], rdects[i][0], rdects[i][1]);

    cv::Matx34d
        lP = read_P(argv[3]),
        rP = read_P(argv[4]);

    std::vector<cv::Vec4d> warped3D;

    cv::triangulatePoints(lP, rP, ldects, rdects, warped3D);
}
