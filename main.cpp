#include "kernel.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cerr << "Usage: ./cuda_vision <image_path>" << std::endl;
		return -1;
	}

	cv::Mat input_full = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	if (input_full.empty()) {
		std::cerr << "Error: Could not open image!" << std::endl;
		return -1;
	}
	cv::Mat output_full = cv::Mat::zeros(input_full.size(), input_full.type());

	// INFO: CUDA kernel
	launch_sobel(input_full.data, output_full.data, input_full.cols,
				 input_full.rows);

	cv::Mat display_in, display_out;
	cv::Size displaySize(600, 600);

	cv::resize(input_full, display_in, displaySize);
	cv::resize(output_full, display_out, displaySize);

	cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Sobel Edge Detection", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("Original", 100, 100);
	cv::moveWindow("Sobel Edge Detection", 700, 100);

	cv::imshow("Original", display_in);
	cv::imshow("Sobel Edge Detection", display_out);

	std::cout << "Windows active. Press 'q' or ESC to exit safely."
			  << std::endl;
	while (true) {
		int key = cv::waitKey(1);
		if (key == 27 || key == 'q')
			break;

		if (cv::getWindowProperty("Original", cv::WND_PROP_VISIBLE) < 1 ||
			cv::getWindowProperty("Sobel Edge Detection",
								  cv::WND_PROP_VISIBLE) < 1) {
			break;
		}
	}

	cv::destroyAllWindows();
	return 0;
}
