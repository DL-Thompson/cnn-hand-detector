xhost +
docker run --gpus all -it --rm -p 8888:8888 --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1 -e DISPLAY=${DISPLAY} --env="QT_X11_NO_MITSHM=1"  -v $(pwd):/home/jovyan/work -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root hand_detector
