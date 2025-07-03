seqlen=( 2097152 )
cp_size=( 8 )
model_size=( "13b")
export profile="false"
export sele_ckpt=0
methods=( "loongtrain" )
LOG_FILE=InternEvo/logs/7b_2048x.log
tee_log=InternEvo/logs/7b_2048x_tee.log
for cp in ${cp_size[@]}; do
  for i in ${seqlen[@]}; do
    for model in ${model_size[@]}; do
      for method in ${methods[@]}; do
        export MODEL_SIZE=$model
        export METHOD=$method
        echo "Running for sequence length: $i and cp size: $cp and model size: $model"
        export SEQ_LEN=$i
        export CP_SIZE=$cp
        export MODEL_SIZE=$model
        export HP_SIZE=$(( $WORLD_SIZE*8 / $CP_SIZE ))
        cmd="bash ./run.sh 2>&1| tee -a $tee_log"
        echo "hpsize $HP_SIZE"
        echo $cmd
        $cmd
        done
    done
  done
done
