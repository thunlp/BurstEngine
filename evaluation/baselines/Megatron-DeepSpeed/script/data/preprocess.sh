python tools/preprocess_data.py \
       --input res.json \
       --output-prefix codeparrot \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --vocab-file vocab.json \
       --merge-file merges.txt \
       --json-keys content \
       --workers 32 \
       --append-eod
