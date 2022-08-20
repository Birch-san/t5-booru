#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# we filter to at least 31 occurrences, in order to get vaporwave 😎
awk -F"\t" -v OFS='\t' \
'$2 && length($2) <= 4 {
  counts[$2] += $1;
}
length($2) > 4 {
  rest = $2;
  while (has_qualifier = match(rest, /^(.*)_\(([^)]*?)\)$/, qualifier_match)) {
    rest = qualifier_match[1];
    qualifier = qualifier_match[2];
    counts[qualifier] += $1;
    if (qualifier == "cosplay") {
      # _(cosplay) qualifies a name label, so what precedes it can be used
      # without any splitting
      counts[rest] += $1;
      next;
    }
  }
  split(rest, parts, /[-_]/);
  for(i in parts) {
    part = parts[i];
    if (part) {
      counts[part] += $1;
    }
  }
}
END {
  for(i in counts) {
    count = counts[i];
    print count, i;
  }
}' \
"$SCRIPT_DIR/../out/general_label_prevalence.raw.tsv" \
| sort -rn \
| awk -F"\t" '$1 >= 31 { print $2 }' \
> "$SCRIPT_DIR/../out/general_tokens.tsv"