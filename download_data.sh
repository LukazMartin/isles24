#!/bin/bash
#
# Download data to session storage
mkdir /home/renku/work/isles2/data-local
#rsync -avz --progress data/train data-local/
cp -r /home/renku/work/s3-isles24/train /home/renku/work/isles2/data-local
