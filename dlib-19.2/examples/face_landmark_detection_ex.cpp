// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.  

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/pixel.h>
#include <dlib/opencv/cv_image.h>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

#include "dlib\all\dlib_test\face_landmark_detection.h"

using namespace dlib;
using namespace std;

class CPerfMeasure
{
	__int64 m_start;
	__int64 m_end;
	
	int m_count;

	double m_fSum;

public:
	CPerfMeasure() {
		m_count = 0;
		m_fSum = 0;
	}
	~CPerfMeasure() {
	}
	void start();
	double end();
	double avg();
};
inline void CPerfMeasure::start() {
	m_start = cv::getTickCount();
}
inline double CPerfMeasure::end() {
	m_end = cv::getTickCount();

	double t = (m_end - m_start) / cv::getTickFrequency();
	m_fSum += t;
	m_count++;

	return avg();
}
inline double CPerfMeasure::avg() {
	return m_fSum / m_count;
}


// ----------------------------------------------------------------------------------------
#if 1
int main(int argc, char** argv)
{  
	CPerfMeasure perf;

	perf.start();

	for (int i = 0; i < 1000; i++) {
		perf.start();

		Sleep(10);

		double fAvg = perf.end();
		cout << fAvg << endl;
	}

	unsigned long startTime = 0;

    try
    {
		dlib::rectangle rc1(10, 10, 100, 100);
		dlib::rectangle rc2(20, 20, 200, 50);
		dlib::rectangle rc3 = rc1 + rc2;
		dlib::rectangle rc4 = rc1.intersect(rc2);

        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.


        frontal_face_detector detector = get_frontal_face_detector();


        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        deserialize(argv[1]) >> sp;

		cv::Mat imageTT = imread(argv[2]);

		dlib::array2d<rgb_pixel> m_cameraFrameGray;
		dlib::assign_image(m_cameraFrameGray, dlib::cv_image<rgb_pixel>(imageTT));


		cv::Mat image = imread(argv[2]);
		startTime = GetTickCount();
		for (int i = 0; i < 1000; i++) {
			cvtColor(image, image, CV_BGR2RGB);
		}

		unsigned long endTime = GetTickCount();
		cout << "same : "<< endTime - startTime << endl;

		cv::Mat image__ = imread(argv[2]);
		cv::Mat image2;
		startTime = GetTickCount();
		for (int i = 0; i < 1000; i++) {
			cvtColor(image__, image2, CV_BGR2RGB);
		}
		endTime = GetTickCount();
		cout << "different " << endTime - startTime << endl;

		startTime = GetTickCount();
		for (int i = 0; i < 5000; i++) {
			flip(image__, image__, 0);
		}
		endTime = GetTickCount();
		cout << "flip same " << endTime - startTime << endl;

		cv::Mat image2_;
		startTime = GetTickCount();
		for (int i = 0; i < 5000; i++) {
			flip(image__, image2_, 0);
		}
		endTime = GetTickCount();
		cout << "flip different " << endTime - startTime << endl;

#if 0
        //image_window win, win_faces;
        // Loop over all the images provided on the command line.
        for (int i = 2; i < argc; ++i)
        {
            cout << "processing image " << argv[i] << endl;
            array2d<rgb_pixel> img;
            load_image(img, argv[i]);

			startTime = GetTickCount();

			for (int j = 0; j < 100; j++) {



            // Make the image larger so we can detect small faces.
//            pyramid_up(img);

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.

			dlib::array2d<unsigned char> faceImg;
			dlib::assign_image(faceImg, img);

#if 1
            std::vector<dlib::rectangle> dets = detector(faceImg);
            cout << "Number of faces detected: " << dets.size() << endl;
#else
			std::vector<rectangle> dets;
			dets.push_back(rectangle(0, 0, 1148, 765));
#endif
			
            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(img, dets[j]);

				//dlib::rectangle rc(0, 0, faceImg.nc()-1, faceImg.nr()-1);
				//full_object_detection shape = sp(img, rc);

                cout << "number of parts: "<< shape.num_parts() << endl;
                cout << "pixel position of first part:  " << shape.part(0) << endl;
                cout << "pixel position of second part: " << shape.part(1) << endl;

				// You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                shapes.push_back(shape);
            }

#if 0

            // Now let's view our face poses on the screen.
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(render_face_detections(shapes));


			// Left eye
			float leye_x = (shapes[0].part(36).x() + shapes[0].part(39).x()) / 2.0f;
			float leye_y = (shapes[0].part(36).y() + shapes[0].part(39).y()) / 2.0f;

			// Right eye
			float reye_x = (shapes[0].part(42).x() + shapes[0].part(45).x()) / 2.0f;
			float reye_y = (shapes[0].part(42).y() + shapes[0].part(45).y()) / 2.0f;

			// nose
			float nose_x = (shapes[0].part(30).x() + shapes[0].part(30).x()) / 2.0f;
			float nose_y = (shapes[0].part(30).y() + shapes[0].part(30).y()) / 2.0f;


			cout << "leye: " << leye_x << "," << leye_y << endl;
			cout << "reye: " << reye_x << "," << reye_y << endl;
			cout << "nose: " << nose_x << "," << nose_y << endl;

			const rgb_pixel color = rgb_pixel(255, 0, 0);
			image_window::overlay_circle leye(point(leye_x, leye_y), 2, color);
			win.add_overlay(leye);

			image_window::overlay_circle reye(point(reye_x, reye_y), 2, color);
			win.add_overlay(reye);

			image_window::overlay_circle nose(point(nose_x, nose_y), 2, color);
			win.add_overlay(nose);
#endif

#if 0
			std::vector<image_window::overlay_line> lines;
			std::vector<point> pt;
			pt.push_back(point(dets[0].left(), dets[0].top()));
			pt.push_back(point(dets[0].right(), dets[0].top()));
			pt.push_back(point(dets[0].right(), dets[0].top()));
			pt.push_back(point(dets[0].right(), dets[0].bottom()));
			pt.push_back(point(dets[0].right(), dets[0].bottom()));
			pt.push_back(point(dets[0].left(), dets[0].bottom()));
			pt.push_back(point(dets[0].left(), dets[0].bottom()));
			pt.push_back(point(dets[0].left(), dets[0].top()));

			for (int m = 0; m < 8; m+=2)
				lines.push_back(image_window::overlay_line(pt[m], pt[m+1], rgb_pixel(255, 0, 0)));

			win.add_overlay(lines);
#endif

			}

            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
			/*
            dlib::array<array2d<rgb_pixel> > face_chips;
            extract_image_chips(img, get_face_chip_details(shapes), face_chips);
            win_faces.set_image(tile_images(face_chips));
			*/

//            cout << "Hit enter to process the next image..." << endl;
//            cin.get();
        }
#endif
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }

	unsigned long endTime = GetTickCount();

	cout << endTime - startTime << endl;
}
#else
int main(int argc, char** argv)
{
//	FLMD_init("D:\\work\\shape_predictor_68_face_landmarks.dat");


	try
	{
		// This example takes in a shape model file and then a list of images to
		// process.  We will take these filenames in as command line arguments.
		// Dlib comes with example images in the examples/faces folder so give
		// those as arguments to this program.
		if (argc == 1)
		{
			cout << "Call this program like this:" << endl;
			cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
			cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
			cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
			return 0;
		}

		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.


		frontal_face_detector detector = get_frontal_face_detector();


		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		shape_predictor sp;
		deserialize(argv[1]) >> sp;


		image_window win, win_faces;
		win.set_size(320, 240);

		int deviceId = 0;

		// Get a handle to the Video device:
		VideoCapture cap(deviceId);
		// Check if we can use this device at all:
		if (!cap.isOpened()) {
			cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
			return -1;
		}

		Mat frame;

		// Loop over all the images provided on the command line.
		for (;;)
		{
			cap >> frame;
			// Clone the current frame:
			Mat original = frame.clone();
/*
			// Convert the current frame to grayscale:
			Mat gray;
			cvtColor(original, gray, CV_BGR2GRAY);
*/

			int width = original.cols;
			int height = original.rows;
			float ratio = height / (float)width;
			
			array2d<rgb_pixel> img;
			dlib::assign_image(img, cv_image<dlib::bgr_pixel>(original));

			array2d<rgb_pixel> img_resize;
			float scale = .2f;
			int w = 160;
			int h = w * ratio;
			img_resize.set_size(h, w);
			dlib::resize_image(img, img_resize);


			// Make the image larger so we can detect small faces.
			pyramid_up(img_resize);
			int w_pyrad = img_resize.nc();
			int h_pyrad = img_resize.nr();

			/*
			for (int mm = 0; mm < h_pyrad; mm++) {
				array2d<rgb_pixel>::row r = img_resize[mm];
				for (int kk = 0; kk < w_pyrad; kk++) {
					rgb_pixel pixel= r[kk];
					pixel.red = 0;
					pixel.green = 0;
					pixel.blue = 0;
				}
			}
			*/

			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces in the image.

			std::vector<dlib::rectangle> dets = detector(img_resize);
			cout << "Number of faces detected: " << dets.size() << endl;

			// Now we will go ask the shape_predictor to tell us the pose of
			// each face we detected.
			std::vector<full_object_detection> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				full_object_detection shape = sp(img_resize, dets[j]);

				cout << "number of parts: " << shape.num_parts() << endl;
				cout << "pixel position of first part:  " << shape.part(0) << endl;
				cout << "pixel position of second part: " << shape.part(1) << endl;

				// You get the idea, you can get all the face part locations if
				// you want them.  Here we just store them in shapes so we can
				// put them on the screen.
				shapes.push_back(shape);
			}


			float w_scale = width / (float)w_pyrad;
			float h_scale = height / (float)h_pyrad;

			std::vector<dlib::rectangle> dets_original;
			for (unsigned long j = 0; j < dets.size(); ++j)
				dets_original.push_back( dlib::rectangle(dets[j].left() * w_scale, dets[j].top() * h_scale, dets[j].right() * w_scale, dets[j].bottom() * h_scale) );


			// Now let's view our face poses on the screen.
			win.clear_overlay();
			win.set_image(img);
			win.add_overlay(render_face_detections(shapes));

			if (shapes.size() > 0) {
				// Left eye
				float leye_x = (shapes[0].part(36).x() + shapes[0].part(39).x()) / 2.0f;
				float leye_y = (shapes[0].part(36).y() + shapes[0].part(39).y()) / 2.0f;

				// Right eye
				float reye_x = (shapes[0].part(42).x() + shapes[0].part(45).x()) / 2.0f;
				float reye_y = (shapes[0].part(42).y() + shapes[0].part(45).y()) / 2.0f;

				// nose
				float nose_x = (shapes[0].part(30).x() + shapes[0].part(30).x()) / 2.0f;
				float nose_y = (shapes[0].part(30).y() + shapes[0].part(30).y()) / 2.0f;


				const rgb_pixel color = rgb_pixel(255, 0, 0);
				image_window::overlay_circle leye(point(leye_x, leye_y), 2, color);
				win.add_overlay(leye);

				image_window::overlay_circle reye(point(reye_x, reye_y), 2, color);
				win.add_overlay(reye);

				image_window::overlay_circle nose(point(nose_x, nose_y), 2, color);
				win.add_overlay(nose);

				std::vector<image_window::overlay_line> lines;
				std::vector<point> pt;
				pt.push_back(point(dets_original[0].left(), dets_original[0].top()));
				pt.push_back(point(dets_original[0].right(), dets_original[0].top()));
				pt.push_back(point(dets_original[0].right(), dets_original[0].top()));
				pt.push_back(point(dets_original[0].right(), dets_original[0].bottom()));
				pt.push_back(point(dets_original[0].right(), dets_original[0].bottom()));
				pt.push_back(point(dets_original[0].left(), dets_original[0].bottom()));
				pt.push_back(point(dets_original[0].left(), dets_original[0].bottom()));
				pt.push_back(point(dets_original[0].left(), dets_original[0].top()));

				for (int m = 0; m < 8; m += 2)
					lines.push_back(image_window::overlay_line(pt[m], pt[m + 1], rgb_pixel(255, 0, 0)));

				win.add_overlay(lines);
			}

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			/*
			dlib::array<array2d<rgb_pixel> > face_chips;
			extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			win_faces.set_image(tile_images(face_chips));
			*/

			char key = (char)waitKey(20);
			// Exit this loop on escape:
			if (key == 27)
				break;
		}
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}

}
#endif

// ----------------------------------------------------------------------------------------

