python tools/preprocess_data.py \
       --input res.json \
       --output-prefix codeparrot \
       --vocab-file vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file merges.txt \
       --json-keys content \
       --workers 32 \
       --append-eod
