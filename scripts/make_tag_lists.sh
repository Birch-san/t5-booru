#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

sqlite3 -batch "$SCRIPT_DIR/../booru-chars.db" >"$SCRIPT_DIR/../out/general_label_prevalence.raw.tsv" <<'SQL'
.mode tabs
select count(*) as prevalence, lower(t.TAG)
from tags t
where t.TAG_CAT = 0
group by lower(t.TAG)
order by prevalence DESC
SQL

sqlite3 -batch "$SCRIPT_DIR/../booru-chars.db" >"$SCRIPT_DIR/../out/name_tokens_prevalence.raw.tsv" <<'SQL'
.mode tabs
select count(*) as prevalence, lower(t.TAG)
from tags t
where t.TAG_CAT != 0
group by lower(t.TAG)
order by prevalence DESC
SQL