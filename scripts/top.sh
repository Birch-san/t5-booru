#!/usr/bin/env zsh
awk -F"\t" -v OFS='\t' '$2 >= 79 {
  counts[tolower($1)] = $2
}
END {
  for(i in counts) {
    count = counts[i];
    print count, i;
  }
}' "$HOME/machine-learning/tokenization/all-non-general-tags-prevalence.tsv" | \
sort -rn | \
awk -F"\t" '$0=$2'