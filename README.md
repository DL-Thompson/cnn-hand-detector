This is a practice project to create a hand detector for my webcam. It uses a Convolutional Neural Network with transfer learning.

To build this project I used JupyterLab. The dockerfile is included, it just needs to be built. After the docker container is built, you can run the container and lab using the run docker bash file. There is one for GPU and no GPU.

To make the data set more manageable to store on Github, the images in the data/original directory and models/final/data have been compressed. To decompress, simply run the untar_data.sh script in the top directory. It will uncompress and then delete the tar files. 

