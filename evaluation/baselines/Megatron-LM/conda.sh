if [ -z $MASTER_ADDR ]; then
  MASTER_ADDR=localhost
fi

if [ "$HOSTNAME" != "$MASTER_ADDR" ]; then
  IS_WORKER=1
fi

export seqlen=$1 
export method=$2 

source activate gcc9 && cd /home/test/test01/sa/workspace/Megatron-LM \
  && bash ./multi.sh
