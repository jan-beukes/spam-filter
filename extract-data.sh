#!/bin/sh

tars=$(ls ./data/enron*.tar.gz)

for tar in $tars; do
    tar xf $tar -C ./data
done

echo Enron data extracted into ./data
