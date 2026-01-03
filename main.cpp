#include "kernel.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "Usage: ./cuda_vision <image_path>" << std::endl;
		return -1;
	}

	cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	if (img.empty())
		return -1;

	cv::Mat output = img.clone();

	// INFO: CUDA kernel
	launch_sobel(img.data, output.data, img.cols, img.rows);

	cv::imshow("Original", img);
	cv::imshow("Sobel Edge Detection", output);
	cv::waitKey(0);

	return 0;
}
