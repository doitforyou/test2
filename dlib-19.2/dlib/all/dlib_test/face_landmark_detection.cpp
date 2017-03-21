#include "face_landmark_detection.h"

using namespace dlib;
using namespace std;

frontal_face_detector g_detector = get_frontal_face_detector();
shape_predictor g_sp;

void FLMD_init(char *shape_predictor_68_face_landmarks_filename)
{
	deserialize(shape_predictor_68_face_landmarks_filename) >> g_sp;
}

int FLMD_detect(array2d<rgb_pixel> &img, float &leye_x, float &leye_y, float &reye_x, float &reye_y, float &nose_x, float &nose_y)
{
	// pyramid_up(img);
	std::vector<rectangle> dets = g_detector(img);
	if (dets.size() <= 0)
		return 0;

	full_object_detection shape = g_sp(img, dets[0]);
	shape.num_parts();

	// Left eye
	leye_x = (shape.part(36).x() + shape.part(39).x()) / 2.0f;
	leye_y = (shape.part(36).y() + shape.part(39).y()) / 2.0f;

	// Right eye
	reye_x = (shape.part(42).x() + shape.part(45).x()) / 2.0f;
	reye_y = (shape.part(42).y() + shape.part(45).y()) / 2.0f;

	// nose
	nose_x = (shape.part(30).x() + shape.part(30).x()) / 2.0f;
	nose_y = (shape.part(30).y() + shape.part(30).y()) / 2.0f;

	return 1;
}

int FLMD_detect(char *face_image_file_name, float &leye_x, float &leye_y, float &reye_x, float &reye_y, float &nose_x, float &nose_y)
{
	array2d<rgb_pixel> img;
	load_image(img, face_image_file_name);

	return FLMD_detect(img, leye_x, leye_y, reye_x, reye_y, nose_x, nose_y);
}
