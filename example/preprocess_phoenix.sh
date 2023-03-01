#!/bin/bash -v

SRC=en
TRG=de

# number of merge operations
bpe_operations=8000
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=
# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=
# path to phoneix 2014T
phoneix=PHOENIX-2014-T-release-v3/PHOENIX-2014-T/
# path to mustc en-de
mustc=en-de/data/train/txt

# extract train/dev/test data from phoenix2014
for prefix in train dev test
do
    python3 extract_phoenix.py \
        ${phoneix}/annotations/manual/PHOENIX-2014-T.${prefix}.corpus.csv \
        ${phoneix}/features/fullFrame-210x260px/ \
        ${prefix}
    cat ${prefix}.src | tr '[:upper:]' '[:lower:]' > ${prefix}.${SRC}.raw
    ln -s ${prefix}.tgt ${prefix}.${TRG}
done

# extract mustc en->de machine translation data
ln -s ${mustc}/train.${SRC} mustc.${SRC}
ln -s ${mustc}/train.${TRG} mustc.${TRG}

# tokenize -> should use sacreBLEU for final evaluation
awk 'BEGIN{i=0} /.*/{printf "%d % s\n",i,$0; i++}' < train.${SRC}.raw > train.${SRC}


# tokenize mustc
for prefix in mustc
do
    cat ${prefix}.${SRC} \
        | ${mosesdecoder}/scripts/tokenizer/normalize-punctuation.perl -l ${SRC} \
        | ${mosesdecoder}/scripts/tokenizer/tokenizer.perl -a -l ${SRC} \
        | tr -d '[:punct:]' | tr '[:upper:]' '[:lower:]' > ${prefix}.tok.${SRC}

    cat ${prefix}.${TRG} \
        | ${mosesdecoder}/scripts/tokenizer/normalize-punctuation.perl -l ${TRG} \
        | ${mosesdecoder}/scripts/tokenizer/tokenizer.perl -a -l ${TRG} \
        | tr -d '[:punct:]' | tr '[:upper:]' '[:lower:]'> ${prefix}.tok.${TRG}
done
# <aug> denotes machine translation
sed -i 's/^/-1 <aug> /g' mustc.tok.${SRC}

# combine must and phoneix
for Lang in ${SRC} ${TRG}
do

# oversampling, 10 times
for i in {1..10}; do cat train.${Lang} >> ${Lang}.txt; done

cat mustc.tok.${Lang} ${Lang}.txt > train.tok.${Lang}

done

cut -d ' ' -f 2- < train.tok.${SRC} > train.tok.${SRC}.no-first
cut -d ' ' -f 1 < train.tok.${SRC} > train.tok.${SRC}.first

# train BPE
cat train.tok.${SRC}.no-first train.tok.${TRG} | ${subword_nmt}/learn_bpe.py -s ${bpe_operations} > ${SRC}${TRG}.bpe

# apply BPE
for prefix in dev test
do
    ${subword_nmt}/apply_bpe.py -c ${SRC}${TRG}.bpe < ${prefix}.${SRC}.raw > ${prefix}.bpe.${SRC}.raw
    awk 'BEGIN{i=0} /.*/{printf "%d % s\n",i,$0; i++}' <  ${prefix}.bpe.${SRC}.raw > ${prefix}.bpe.${SRC}
    ${subword_nmt}/apply_bpe.py -c ${SRC}${TRG}.bpe < ${prefix}.${TRG} > ${prefix}.bpe.${TRG}
done

# apply BPE to the training data
prefix=train
${subword_nmt}/apply_bpe.py -c ${SRC}${TRG}.bpe < ${prefix}.tok.${SRC}.no-first > ${prefix}.bpe.${SRC}.no-first
${subword_nmt}/apply_bpe.py -c ${SRC}${TRG}.bpe < ${prefix}.tok.${TRG} > ${prefix}.bpe.${TRG}
paste -d ' ' ${prefix}.tok.${SRC}.first ${prefix}.bpe.${SRC}.no-first > train.bpe.en

# shuffle training data
python shuffle_corpus.py --corpus train.bpe.${SRC} train.bpe.${TRG}
cat ${prefix}.bpe.${SRC}.no-first train.bpe.${TRG} > tmp

# extract vocabulary
python vocab.py tmp vocab.zero
python bpe_vocab_fuse.py ${SRC}${TRG}.bpe vocab.zero vocab.zero.drop
rm tmp


# outputs explanation
# - train/dev/test.src <= gloss file
# - train/dev/test.tgt <= translation file
# - train/dev/test.img <= sign video path
# - vocab.zero.drop <= vocabulary considering bpe-dropout
# - train/dev/test.bpe.en/de <= bpe tokenized files
# - train.bpe.en/de.shuf <= randomly shuffled training data, used for training
