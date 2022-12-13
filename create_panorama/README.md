# create panorama
Establish the correspondence between images using SIFT and reject the outliers with
RANSAC algorithm. Estimate homography using linear least square and refine it using
non-linear least square (LM). Use the homographies to stitch the images and form a
panoramic view.