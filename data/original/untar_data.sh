#!/bin/bash

echo 'Uncompressing directories...'
for file in *.tar.gz; do
    tar -xzf $file
    rm $file
done
echo 'Uncompression complete.'