#!/bin/bash

#tar data in the data/original folder
echo 'Compressing /data/original/ directories...'
cd data/original
for dir in */; do
    tar -czf ${dir::-1}.tar.gz ./$dir
    rm -r $dir
done
cd ../../
echo 'Compression of /data/original/ complete.'

#tar data in models/final/data
cd models/final/data
echo 'Compressing /models/final/data/ directories...'
for dir in */; do
    if [ $dir == "train/" ]; then
        total=$(ls ./train/positive | wc -l)
        half=$((total / 2))

        mkdir train_1 #first half of positive
        mkdir train_2 #second half of positive
        mkdir train_3 #negative

        i=0
        for file in ./train/positive/*; do
            #file=${file:17}
            if [ $i -le $half ]; then
                #echo train_1/$file
                cp $file ./train_1
            fi
            if [ $i -gt $half ]; then
                #echo train_2/$file
                cp $file ./train_2
            fi
            i=$((i+1))
        done

        cp ./train/negative/* train_3
        tar -czf train_1.tar.gz ./train_1
        tar -czf train_2.tar.gz ./train_2
        tar -czf train_3.tar.gz ./train_3
        rm -r train train_1 train_2 train_3
    else
        tar -czf ${dir::-1}.tar.gz ./$dir
        rm -r $dir
    fi
done
cd ../../../
echo 'Compression of /models/final/data/ complete.'