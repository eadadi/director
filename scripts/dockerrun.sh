#!/bin/bash

# Launch from director main folder
docker run --rm -it --network host \
	-v $(pwd):/code \
	-v /home/ec2-user/drive/project/logdir:/logdir \
	-w /code \
	--gpus all \
	mydirector bash
