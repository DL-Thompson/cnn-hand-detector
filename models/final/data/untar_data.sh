#!/bin/bash

echo 'Uncompressing directories...'
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

echo 'Uncompression complete.'