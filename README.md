## Cudacv

GPU accelarated Image preprocessing pipeline

### Build

```sh
mkdir build && cd build
cmake ..
make -j$(nproc)
./cuda_vision path/to/your/image.jpg
```
<table> <tr> <th>Original</th> <th>Sobel Edge Detection (CUDA)</th> </tr> <tr> <td> <img src="https://github.com/user-attachments/assets/c5806a39-2fdb-4242-8426-ef79b8cc52ec" width="450" /> </td> <td> <img src="https://github.com/user-attachments/assets/52b79b18-1c99-41e7-8167-2bce18f07dcf" width="450" /> </td> </tr> </table>
