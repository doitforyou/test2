#pragma once

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

void FLMD_init(char *shape_predictor_68_face_landmarks_filename);
int FLMD_detect(dlib::array2d<dlib::rgb_pixel> &img, float &leye_x, float &leye_y, float &reye_x, float &reye_y, float &nose_x, float &nose_y);
int FLMD_detect(char *face_image_file_name, float &leye_x, float &leye_y, float &reye_x, float &reye_y, float &nose_x, float &nose_y);
