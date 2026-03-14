#!/bin/bash
#
# Upload all runs to remote storage
#mkdir /home/renku/work/data/runs/
#rsync -avz --progress /home/renku/work/data-local/runs/ /home/renku/work/data/runs/
cp -r /home/renku/work/isles24/data-local/runs /home/renku/work/s3-isles24/
