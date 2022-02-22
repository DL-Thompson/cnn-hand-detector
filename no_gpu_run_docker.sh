docker run -it --rm -p 8888:8888 -v $(pwd):/home/jovyan/work -e GRANT_SUDO=yes -e JUPYTER_ENABLE_LAB=yes --user root hand_detector
