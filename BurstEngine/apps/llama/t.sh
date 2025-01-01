nnodes=1
gpus=8
tp=$(( $nnodes * $gpus ))
echo $tp
