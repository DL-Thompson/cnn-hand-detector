#!/bin/bash

#untar data in the data/original folder
echo 'Uncompressing /data/original/ directories...'
cd data/original
for file in *.tar.gz; do
    tar -xzf $file
    rm $file
done
cd ../../
echo 'Uncompression of /data/original/ complete.'

#untar data in models/final/data
cd models/final/data
echo 'Uncompressing /models/final/data/ directories...'

for file in *.tar.gz; do
    tar -xzf $file
    rm $file
done

mkdir -p train/positive
mv train_1/* ./train/positive
mv train_2/* ./train/positive

mkdir -p train/negative
mv train_3/* ./train/negative
rm -r train_1 train_2 train_3

cd ../../../
echo 'Uncompression of /models/final/data/ complete.'