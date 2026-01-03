#include "kernel.h"
#include <chrono>
#include <filesystem>
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
		std::cerr << "Usage: " << argv[0] << " <path_to_image_or_video>\n";
		return 1;
	}

	std::string path = argv[1];
	std::string ext = std::filesystem::path(path).extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	bool isImage = (ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
					ext == ".bmp" || ext == ".webp");

	cv::VideoCapture cap;
	cv::Mat staticImg;
	int width, height;

	if (isImage) {
		staticImg = cv::imread(path);
		if (staticImg.empty()) {
			std::cerr << "Error: Could not open image " << path << std::endl;
			return -1;
		}
		width = staticImg.cols;
		height = staticImg.rows;
	} else {
		cap.open(path);
		if (!cap.isOpened()) {
			std::cerr << "Error: Could not open video " << path << std::endl;
			return -1;
		}
		width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	}

	size_t graySize = width * height * sizeof(unsigned char);
	unsigned char *d_in, *d_out;
	allocate_buffers(&d_in, &d_out, graySize);

	cv::Mat frame, grayFrame, outputFrame_Gray, outputFrame_Final;
	outputFrame_Gray = cv::Mat::zeros(height, width, CV_8UC1);

	cv::namedWindow("CUDA Pipeline", cv::WINDOW_NORMAL);
	cv::resizeWindow("CUDA Pipeline", 1280, 720);

	int currentMode = MODE_GRAYSCALE;
	int threshold = 200;
	float strength = 1.0f;

	while (true) {
		if (isImage) {
			frame = staticImg.clone();
		} else {
			cap >> frame;
			if (frame.empty()) {
				cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Loop video
				continue;
			}
		}

		if (currentMode == MODE_ORIGINAL) {
			outputFrame_Final = frame;
		} else {
			cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
			upload_to_gpu(d_in, grayFrame.data, graySize);

			auto start = std::chrono::high_resolution_clock::now();

			if (currentMode == MODE_SOBEL)
				launch_sobel_exec(d_in, d_out, width, height, threshold);
			else if (currentMode == MODE_BLUR)
				launch_blur_exec(d_in, d_out, width, height, threshold,
								 strength);
			else if (currentMode == MODE_INVERT)
				launch_invert_exec(d_in, d_out, width, height);
			else
				launch_copy_exec(d_in, d_out, width, height);

			download_from_gpu(outputFrame_Gray.data, d_out, graySize);

			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> ms = end - start;

			cv::cvtColor(outputFrame_Gray, outputFrame_Final,
						 cv::COLOR_GRAY2BGR);

			cv::putText(outputFrame_Final, std::to_string(ms.count()) + " ms",
						cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 1.5,
						cv::Scalar(0, 255, 0), 3);
		}

		std::string modeNames[] = {"ORIGINAL (RGB)", "GRAYSCALE", "SOBEL",
								   "INVERT", "GAUSSIAN BLUR"};
		cv::putText(outputFrame_Final, "MODE: " + modeNames[currentMode],
					cv::Point(50, 70), cv::FONT_HERSHEY_SIMPLEX, 2.0,
					cv::Scalar(0, 255, 255), 4);
		if (currentMode == MODE_SOBEL) {
			cv::putText(outputFrame_Final,
						"THRESHOLD: " + std::to_string(threshold),
						cv::Point(50, 220), cv::FONT_HERSHEY_SIMPLEX, 1.5,
						cv::Scalar(255, 0, 255), 3);
		}
		if (currentMode == MODE_BLUR) {
			cv::putText(outputFrame_Final,
						"STRENGTH: " + std::to_string(strength),
						cv::Point(50, 220), cv::FONT_HERSHEY_SIMPLEX, 1.5,
						cv::Scalar(255, 50, 155), 3);
		}

		cv::imshow("CUDA Pipeline", outputFrame_Final);

		char key = (char)cv::waitKey(isImage ? 0 : 1);
		if (key == 'w')
			threshold = std::min(threshold + 5, 255);
		else if (key == 's')
			threshold = std::max(threshold - 5, 0);
		else if (key == 'a')
			strength = std::fmaxf(strength - 0.1f, 1.0f);
		else if (key == 'd')
			strength = std::fminf(strength + 0.1f, 5.0f);
		if (key == 'q' || key == 27)
			break;
		if (key >= '0' && key <= '4')
			currentMode = (PipelineMode)(key - '0');
	}

	free_buffers(d_in, d_out);
	return 0;
}
