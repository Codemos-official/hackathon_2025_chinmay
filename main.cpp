#include "kernel.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

enum PipelineMode {
	MODE_ORIGINAL = 0,
	MODE_GRAYSCALE = 1,
	MODE_SOBEL = 2,
	MODE_INVERT = 3,
	MODE_BLUR = 4
};

int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <video_path>\n";
		return 1;
	}

	std::string videoPath = argv[1];
	cv::VideoCapture cap(videoPath);

	if (!cap.isOpened()) {
		std::cerr << "Error: Could not open video file at " << videoPath
				  << std::endl;
		return -1;
	}

	int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	size_t graySize = width * height * sizeof(unsigned char);

	std::cout << "Processing 4K Video: " << width << "x" << height << std::endl;

	unsigned char *d_in, *d_out;
	allocate_buffers(&d_in, &d_out, graySize);

	cv::Mat frame, grayFrame, outputFrame_Gray, outputFrame_Final;
	outputFrame_Gray = cv::Mat::zeros(height, width, CV_8UC1);

	cv::namedWindow("CUDA 4K Pipeline", cv::WINDOW_NORMAL);
	cv::resizeWindow("CUDA 4K Pipeline", 1280, 720);

	int currentMode = MODE_GRAYSCALE;

	while (true) {
		cap >> frame;
		if (frame.empty()) {
			cap.set(cv::CAP_PROP_POS_FRAMES, 0);
			continue;
		}

		if (currentMode == MODE_ORIGINAL) {
			outputFrame_Final = frame;
		} else {
			cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
			upload_to_gpu(d_in, grayFrame.data, graySize);

			auto start = std::chrono::high_resolution_clock::now();

			if (currentMode == MODE_SOBEL) {
				launch_sobel_exec(d_in, d_out, width, height);
			} else if (currentMode == MODE_BLUR) {
				launch_blur_exec(d_in, d_out, width, height);
			} else if (currentMode == MODE_INVERT) {
				launch_invert_exec(d_in, d_out, width, height);
			} else {
				launch_copy_exec(d_in, d_out, width, height);
			}

			download_from_gpu(outputFrame_Gray.data, d_out, graySize);

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> ms = end - start;

			cv::cvtColor(outputFrame_Gray, outputFrame_Final,
						 cv::COLOR_GRAY2BGR);

			cv::putText(outputFrame_Final, std::to_string(ms.count()) + " ms",
						cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 2.0,
						cv::Scalar(0, 255, 0), 4);
		}

		std::string modeNames[] = {"ORIGINAL (RGB)", "GRAYSCALE", "SOBEL",
								   "INVERT", "GAUSSIAN BLUR"};
		cv::putText(outputFrame_Final, "MODE: " + modeNames[currentMode],
					cv::Point(50, 70), cv::FONT_HERSHEY_SIMPLEX, 2.0,
					cv::Scalar(0, 255, 255), 4);

		cv::imshow("CUDA 4K Pipeline", outputFrame_Final);

		char key = (char)cv::waitKey(1);
		if (key == 'q')
			break;
		if (key >= '0' && key <= '4')
			currentMode = (PipelineMode)(key - '0');
	}

	free_buffers(d_in, d_out);
	return 0;
}
