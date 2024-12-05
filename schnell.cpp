#include <iostream>
#include <algorithm>
#include <print>
#include <set>

#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

image_u8_t april_from_mat(const cv::Mat_<uint8_t>& m) {
    return (image_u8_t) {
        .width  = m.cols,
        .height = m.rows,
        .stride = m.step,
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

void intersect_dectvecs(
    std::vector<apriltag_detection_t> &l,
    std::vector<apriltag_detection_t> &r,
    std::vector<cv::Vec2d> &lout,
    std::vector<cv::Vec2d> &rout
) {
    std::vector<int> lids, rids;
    std::set<int> isect;

    auto idcomp = [](const auto& l, const auto& r) { return l.id < r.id; };

    std::ranges::sort(l, idcomp);
    std::ranges::sort(r, idcomp);

    auto map = [](const auto &d){ return d.id; };

    std::ranges::transform(l, std::back_inserter(lids), map);
    std::ranges::transform(r, std::back_inserter(rids), map);

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

void detect_union(
    const char *lpath,
    const char *rpath,
    std::vector<cv::Vec2d> &ldects,
    std::vector<cv::Vec2d> &rdects    
) {
    cv::Mat_<uint8_t>
        limg = cv::imread(lpath, cv::IMREAD_GRAYSCALE),
        rimg = cv::imread(rpath, cv::IMREAD_GRAYSCALE);

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

    intersect_dectvecs(ld, rd, ldects, rdects);
}

cv::Matx34d read_P(const char *path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);

    cv::Matx34d P;
    fs["Q"] >> P;

    fs.release();

    return P;
}

int main(int argc, const char **argv) {

    if (argc != 5) {
        std::println(std::cerr, "Usage: {} [limg] [rimg] [lcal] [rcal]", argv[0]);
        return 1;
    }

    std::vector<cv::Vec2d> ldects, rdects;
    detect_union(argv[1], argv[2], ldects, rdects);

    for (size_t i = 0; i < ldects.size(); ++i)
        std::println("({}, {}) | ({}, {})", ldects[i][0], ldects[i][1], rdects[i][0], rdects[i][1]);

    cv::Matx34d
        lP = read_P(argv[3]),
        rP = read_P(argv[4]);

    std::vector<cv::Vec4d> warped3D;

    cv::triangulatePoints(lP, rP, ldects, rdects, warped3D);
}
