#!/bin/bash
FILES=images/*
for f in $FILES
do
  for g in $FILES
    do
      python neuralNet.py $f $g
    done
done