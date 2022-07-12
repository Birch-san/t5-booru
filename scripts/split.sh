#!/usr/bin/env zsh
cat <(awk -F"\t" '$2 > 33' "$HOME/machine-learning/tokenization/all-general-tags-prevalence.tsv") \
"$HOME/git/walfie-gif-dl/out/general_tags_manual.tsv" | \
awk -F"\t" -v OFS='\t' 'length($1) <= 4 {
  counts[tolower($1)] += $2;
}
length($1) > 4 {
  split(tolower($1), parts, /[-_]/);
  for(i in parts) {
    part = parts[i];
    counts[part] += $2;
  }
}
END {
  for(i in counts) {
    count = counts[i];
    print count, i;
  }
}' | sort -rn | \
awk -F"\t" '$0=$2'