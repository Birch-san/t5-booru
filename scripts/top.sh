#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# we filter to at least 83 occurrences, in order to get amano_pikamee ⚡️
awk -F"\t" -v OFS='\t' \
'$1 >= 83 { print $2 }' \
"$SCRIPT_DIR/../out/name_tokens_prevalence.raw.tsv" \
> "$SCRIPT_DIR/../out/label_tokens.tsv"