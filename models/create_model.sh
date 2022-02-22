#!/bin/bash

max_model=$(ls -d */ | sort -n | tail -1)
max_model=${max_model%/}
model_num=$((10#$max_model + 1))
prev_model=$((10#$max_model))

format_number () {
    if [ ${#1} == 1 ]
    then
        local num='000'$1
    elif [ ${#1} == 2 ]
    then
        local num='00'$1
    elif [ ${#1} == 3 ]
    then
        local num='0'$1
    fi
    echo $num
}

model_num=$(format_number $model_num)
prev_model=$(format_number $prev_model)

if [ -d $model_num ]
then
    echo 'Model number already exists.'
    exit 1
fi

echo 'Making new directory for model: ' $model_num
mkdir $model_num

echo 'Creating subdirectories: '
echo ' '$model_num/data
echo ' '$model_num/data/train
echo ' '$model_num/data/test
echo ' '$model_num/data/validation
echo ' '$model_num/saves
mkdir $model_num/data
mkdir $model_num/data/train
mkdir $model_num/data/test
mkdir $model_num/data/validation
mkdir $model_num/saves

echo 'Copying load_data.ipynb to: ' ./$model_num/data
cp ./$prev_model/data/load_data.ipynb ./$model_num/data
cp ./$prev_model/data/load_positive_data.ipynb ./$model_num/data

echo 'Copying model.ipynb to: ' ./$model_num
cp ./$prev_model/model.ipynb ./$model_num

echo 'Copying util files to: ' ./$model_num
cp ./$prev_model/project_utils*.py ./$model_num
