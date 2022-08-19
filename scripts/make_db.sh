#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

rm "$SCRIPT_DIR/../booru-chars.db"
exec sqlite3 -batch "$SCRIPT_DIR/../booru-chars.db" <<'SQL'
.mode tabs
create table files (
  booru text not null,
  fid integer not null,
  shard text not null,
  file_name text not null,
  torr_md5 text,
  orig_ext text,
  orig_md5 text,
  file_size integer,
  img_width_torr integer,
  img_height_torr integer,
  jq text,
  torr_path text,
  tags_copyr text,
  tags_char text,
  tags_artist text,
  primary key (booru, fid)
);
create index idx_file_name on files (file_name);
create table tags (
  booru text,
  fid integer,
  tag text,
  tag_id integer,
  tag_cat integer,
  danb_fr text,
  primary key (booru, fid, tag),
  foreign key (booru, fid)
    references files (booru, fid)
    on delete cascade
    on update no action
);
create index idx_tag on tags (tag);
create index idx_tag_cat on tags (tag_cat);
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021a\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2021a/V2021A_files.tsv"' files
.import --skip 1 '|grep -v -e "\"" "$HOME/machine-learning/booru-chars/Safebooru 2021a/V2021A_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021b\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2021b/V2021B_files.tsv"' files
.import --skip 1 '|grep -v -e "\"" "$HOME/machine-learning/booru-chars/Safebooru 2021b/V2021B_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021c\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2021c/V2021C_files.tsv"' files
.import --skip 1 '|uniq "$HOME/machine-learning/booru-chars/Safebooru 2021c/V2021C_tags.tsv" | grep -v -e "\""' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021d\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2021d/V2021D_files.tsv"' files
.import --skip 1 '|grep -v -e "\"" "$HOME/machine-learning/booru-chars/Safebooru 2021d/V2021D_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2022a\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2022a/V2022A_files.tsv"' files
.import --skip 1 '|grep -v -e "\"" "$HOME/machine-learning/booru-chars/Safebooru 2022a/V2022A_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2022b\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2022b/V2022B_files.tsv"' files
.import --skip 1 '|grep -v -e "\"" "$HOME/machine-learning/booru-chars/Safebooru 2022b/V2022B_tags.tsv"' tags
SQL