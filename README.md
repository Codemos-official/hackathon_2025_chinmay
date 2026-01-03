## CudaCV

GPU-accelerated image and video preprocessing pipeline built with CUDA and OpenCV.  
Designed for real-time experimentation with classic vision operators such as Gaussian blur and Sobel edge detection.

## Features

- CUDA-accelerated Sobel edge detection
- Gaussian blur with adjustable strength and threshold
- Supports **both images and videos**
- Real-time parameter tuning via keyboard controls
- Modular pipeline with multiple processing modes

## Build

```sh
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```sh
./cuda_vision <path_to_image_or_video>
```

If a video is provided, frames are processed in real time.
If an image is provided, the pipeline runs in interactive mode.

## Controls

| Key         | Action                          |
| ----------- | ------------------------------- |
| `w`         | Increase Sobel threshold        |
| `s`         | Decrease Sobel threshold        |
| `d`         | Increase Gaussian blur strength |
| `a`         | Decrease Gaussian blur strength |
| `0–4`       | Switch pipeline mode            |
| `q` / `Esc` | Quit                            |

## Pipeline Preview

**Original vs CUDA Sobel Edge Detection**

<table>
  <tr>
    <th>Original</th>
    <th>Sobel Edge Detection (CUDA)</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/c5806a39-2fdb-4242-8426-ef79b8cc52ec" width="450"/>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/52b79b18-1c99-41e7-8167-2bce18f07dcf" width="450"/>
    </td>
  </tr>
</table>

## Vision Pipeline Demo

[https://github.com/user-attachments/assets/bf129cb7-073c-4c1e-9959-d3429dc253b7](https://github.com/user-attachments/assets/bf129cb7-073c-4c1e-9959-d3429dc253b7)

## Notes

- Optimized for NVIDIA GPUs with CUDA support
- Intended for learning, experimentation, and prototyping GPU vision pipelines
- Designed to be easily extensible with additional CUDA kernels
