## Cudacv

GPU accelarated Image preprocessing pipeline

### Build

```sh
mkdir build && cd build
cmake ..
make -j$(nproc)
./cuda_vision path/to/your/image.jpg
```
