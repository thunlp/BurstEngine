

#!/bin/bash


export NODES="g41 g47 g48 g71"
export NODES="g41 g47 g71 g74"
export NODES="g41 g47 g48 g49 g70 g72 g73 g74"
# export NODES="g41"
DOCKER_DIR=InternEvo
export WORLD_SIZE=8
export LOG_FILE=$DOCKER_DIR/logs/${WORLD_SIZE}_nodes/
export LOG_FILE=${LOG_FILE}/`date +%Y%m%d_%H%M%S`.log
echo $LOG_FILE
    bash $PROJECT_DIR/submit.sh "bash build_run.sh" 
    # bash $PROJECT_DIR/submit.sh "bash conda.sh $size $method" 
    done
  done
done 
