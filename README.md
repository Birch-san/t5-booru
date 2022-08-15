# t5-booru

Trying here to fine-tune an off-the-shelf T5 checkpoint on tag sets from Danbooru images, to understand (for example) that `usada_pekora` implies `bunny_ears` and `carrot`. The hope is also that knowledge T5 learned in pre-training (from the C4 corpus) can hydrate our Danbooru tags with more implications (e.g. that they're human).

## Setup repo

```bash
python3 -m venv venv
. ./venv/bin/activate
pip install --upgrade pip
pip install datasets transformers sentencepiece accelerate tokenizers flax wheel jax optax more_itertools lightning
pip install --pre "torch>1.13.0.dev20220610" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install wandb
# for Jupyter notebooks
pip install ipython jupyter ipywidgets ipykernel

wandb login
```

## Make a database of Danbooru tags, from the booru-chars distribution

Download a recent booru-chars distribution, e.g. https://nyaa.si/view/1486179.  
You can download with a bittorrent client such as [qBittorrent](https://www.qbittorrent.org/download.php).  
Download from booru-chars distributions the `*_tags.tsv` and `*_files.tsv`.

### Create & populate database

Construct a sqlite database.

```bash
sqlite3 -batch booru-chars.db <<'SQL'
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
  primary key(booru, fid)
);
create index idx_file_name on files (file_name);
create table tags (
  booru text,
  fid integer,
  tag text,
  tag_id integer,
  tag_cat integer,
  danb_fr text,
  foreign key (booru, fid)
    references files (booru, fid)
    on delete cascade
    on update no action
);
create index idx_tag on tags (tag);
create index idx_tag_cat on tags (tag_cat);
create index idx_booru_fid on tags (booru, fid);
.read ./queries/walfie.sql
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021a\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2021a/V2021A_files.tsv"' files
.import --skip 1 '|cat "$HOME/machine-learning/booru-chars/Safebooru 2021a/V2021A_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021a\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2021b/V2021B_files.tsv"' files
.import --skip 1 '|cat "$HOME/machine-learning/booru-chars/Safebooru 2021b/V2021B_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021a\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2021c/V2021C_files.tsv"' files
.import --skip 1 '|cat "$HOME/machine-learning/booru-chars/Safebooru 2021c/V2021C_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021a\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2021d/V2021D_files.tsv"' files
.import --skip 1 '|cat "$HOME/machine-learning/booru-chars/Safebooru 2021d/V2021D_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021a\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2022a/V2022A_files.tsv"' files
.import --skip 1 '|cat "$HOME/machine-learning/booru-chars/Safebooru 2022a/V2022A_tags.tsv"' tags
.import '|awk -F"\t" "NR>1 && ! /\"/ { for(i=1; i<=2; i++) { printf \"%s\t\", \$i; } printf \"Safebooru 2021a\t\"; for(i=3; i<=7; i++) { printf \"%s\t\", \$i; } split(\$8, size_parts, \"x\"); printf \"%d\t%d\t\", size_parts[1], size_parts[2];  for(i=9; i<=12; i++) { printf \"%s\t\", \$i; } printf \"%s\n\", \$13; }" "$HOME/machine-learning/booru-chars/Safebooru 2022b/V2022B_files.tsv"' files
.import --skip 1 '|cat "$HOME/machine-learning/booru-chars/Safebooru 2022b/V2022B_tags.tsv"' tags
SQL
```
