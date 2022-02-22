#!/bin/bash

echo Zipping files for model: $1
num=$(ls ./$1/data/train | wc -l)
echo Zipping $num in ./$1/data/train
num=$(ls ./$1/data/validation | wc -l)
echo Zipping $num in ./$1/data/validation
num=$(ls ./$1/data/test | wc -l)
echo Zipping $num in ./$1/data/test
tar -C ./$1/data -czf data_set.tar.gz train validation test




