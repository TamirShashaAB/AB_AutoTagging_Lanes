#!/bin/bash

S3_URI=s3://b2b-datasets/sets_to_generate_gt_for/IL_weekend_run/64eef8f8ac243656c2bd2b76/images/
LOCAL_PATH=/home/tamir/datasets/IL_weekend_run/64eef8f8ac243656c2bd2b76/

aws s3 cp ${S3_URI} ${LOCAL_PATH} --recursive