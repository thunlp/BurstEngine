

#!/bin/bash

# sizes=( 524288 )
sizes=( 65536 131072 262144 524288 )

export NODES="g41 g47 g48 g71"
export NODES="g41 g47 g74 g75 g45 g49 g72"
DOCKER_DIR=InternEvo
export WORLD_SIZE=8
export LOG_FILE=$DOCKER_DIR/70b_loongtrain-128-512k.log
export MODEL="30b"
echo $LOG_FILE
export HP_SIZE=1
sizes=( 65536 )
for sele_ckpt in 0; do
for size in ${sizes[@]}; do
  for whole_cp in 2; do
    if [[ $size -eq 262144 && $whole_cp -eq 16 ]]; then
      continue
    fi
    if [[ $size -eq 524288 && ($whole_cp -eq 8 || $whole_cp -eq 16) ]]; then
      continue
    fi
    /home/test/test01/sa/bin/zellij action rename-pane "loongtrain seqlen-$size whole_cp-$whole_cp"
    export CP_SIZE=$(($whole_cp / $HP_SIZE))
    export sele_ckpt=$sele_ckpt
    echo "Running cp $cp hp $hp size $size with ckpt $sele_ckpt" >> summary.txt
    bash $PROJECT_DIR/submit.sh "bash build_run.sh $size" 
    # bash $PROJECT_DIR/submit.sh "bash conda.sh $size $method" 
    done
  done
done 
