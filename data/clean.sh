#!/bin/bash

echo "Cleaning data..."
bzcat cdli_atf_20180602.txt.bz2 \
 | grep "^[0-9]" \
 | sed 's/^[^ ]*//g' \
 | tr -d "|,#\!?[]<>" \
 | sed 's/ \+/ /g' \
 | tr "a-z" "A-Z" \
 | tr "~" "-" \
 > variants.txt

echo "Replacing X and ... with UNK..."
sed -i 's/\(+\| \)X/\1UNK/g' variants.txt
sed -i 's/\.\.\./UNK/g' variants.txt
sed -i 's/MXXX/UNK/g' variants.txt

cp variants.txt variants.prune_more.txt
cp variants.txt variants.prune_less.txt

echo "Creating file without sign variants..."
cat variants.txt |sed 's/-[a-zA-Z0-9]*//g' > novariants.prune_more.txt
cp novariants.prune_more.txt novariants.prune_less.txt

threshold=3

echo "Replacing rare (whole) words with UNK..."
for n in $( seq $threshold ); do
  for file in {,no}variants.prune_more.txt; do
    echo "> $file $n/$threshold"
    for word in $( cat $file |tr " " "\n" |sort |uniq -c |grep " $n " |awk '{print $2}' ); do
      # Replace rare word wherever it occurs (incl. in compounds):
      # sed -i 's/\( \|+\)'$word'\( \|+\)/\1UNK\2/g' clean.txt
      # Replace only whole words:
      sed -i 's/ '$word' / UNK /g' $file
    done
  done
done

echo "Replacing rare words (incl. in compounds) with UNK..."
for n in $( seq $threshold ); do
  for file in {,no}variants.prune_less.txt; do
    echo "> $file $n/$threshold"
    for word in $( cat $file |tr "+" " " |tr " " "\n" |sort |uniq -c |grep " $n " |awk '{print $2}' ); do
      # Replace rare word wherever it occurs (incl. in compounds):
      # sed -i 's/\( \|+\)'$word'\( \|+\)/\1UNK\2/g' clean.txt
      # Replace only whole words:
      sed -i 's/ '$word' / UNK /g' $file
    done
  done
done

echo "Making train/dev/test split..."
for file in {,no}variants.prune_{more,less}.txt; do
  head -n 500 $file > ${file/txt/valid}
  head -n 1000 $file |tail -n 500 > ${file/txt/test}
  tail -n +1000 $file > ${file/txt/train}
done
