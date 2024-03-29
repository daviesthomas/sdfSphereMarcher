# sphereMarcher

A tool for sphere marching over mlps trained to produce sdf values given point queries... Makes pretty renders!

## Installation
First clone repo (and all dependencies)

    git clone --recursive https://github.com/daviesthomas/sdfSphereMarcher.git
    cd sdfSphereMarcher
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j 8

## Usage

    ./sdfViewer_bin 
        -H image height
        -W image width
        -i input mesh or weights
        -o output image path
        -s shading type (0: outline, 1: grayscale, 2: phong)

render cheburashka weights onto a 512x512 image with phong shading! (example seen below)

    ./sdfViewer_bin -H 512 -W 512 -i ../examples/weights/cheburashka.h5 -o ../examples/images/cheburashka.png -s 2

render cheburashka mesh for comparison :)

    ./sdfViewer_bin -H 512 -W 512 -i ../examples/mesh/cheburashka.obj -o ../examples/images/cheburashka-real.png -s 2

## Example Images!

### Phong rendering
![alt text](https://github.com/daviesthomas/sdfSphereMarcher/blob/master/examples/images/armadillo.png)
![alt text](https://github.com/daviesthomas/sdfSphereMarcher/blob/master/examples/images/arm.png)
![alt text](https://github.com/daviesthomas/sdfSphereMarcher/blob/master/examples/images/cow.png)
![alt text](https://github.com/daviesthomas/sdfSphereMarcher/blob/master/examples/images/horse.png)

### Gray Rendering
![alt text](https://github.com/daviesthomas/sdfSphereMarcher/blob/master/examples/images/armadillo-grey.png)
![alt text](https://github.com/daviesthomas/sdfSphereMarcher/blob/master/examples/images/arm-grey.png)
![alt text](https://github.com/daviesthomas/sdfSphereMarcher/blob/master/examples/images/cow-grey.png)
![alt text](https://github.com/daviesthomas/sdfSphereMarcher/blob/master/examples/images/horse-grey.png)
