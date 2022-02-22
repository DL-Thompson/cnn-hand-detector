#!/bin/bash

echo 'Compressing directories...'
for dir in */; do
    tar -czf ${dir::-1}.tar.gz ./$dir
    rm -r $dir
done
echo 'Compression complete.'