#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# we filter to at least 31 occurrences, in order to get vaporwave 😎
awk -F"\t" -v OFS='\t' \
'$2 && length($2) <= 4 {
  counts[$2] += $1;
}
length($2) > 4 {
  if ($2 ~ /_\(cosplay\)$/) {
    cosplay_count++;
    next;
  }
  split($2, parts, /[-_]/);
  for(i in parts) {
    part = parts[i];
    if (part) {
      counts[part] += $1;
    }
  }
}
END {
  if (cosplay_count) {
    print cosplay_count, "(cosplay)";
  }
  for(i in counts) {
    count = counts[i];
    print count, i;
  }
}' \
"$SCRIPT_DIR/../out/general_label_prevalence.raw.tsv" \
| sort -rn \
| awk -F"\t" '$1 >= 31 { print $2 }' \
> "$SCRIPT_DIR/../out/general_tokens.tsv"