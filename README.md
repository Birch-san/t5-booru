# t5-booru

Trying here to fine-tune an off-the-shelf T5 checkpoint on tag sets from Danbooru images, to understand (for example) that `usada_pekora` implies `bunny_ears` and `carrot`. The hope is also that knowledge T5 learned in pre-training (from the C4 corpus) can hydrate our Danbooru tags with more implications (e.g. that they're human).

## Setup repo

```bash
python3 -m venv venv
. ./venv/bin/activate
pip install --upgrade pip
pip install datasets transformers sentencepiece accelerate tokenizers flax wheel jax optax
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

### Sanitize raw inputs

`V2022B_tags.tsv` make contain lines that have double-quotes, like this:

```
www.zerochan.net	3633380	"I'm_One_Dapping_Fella"_S...	0	3	COPYRIGHT
```

Remove them like so:

```bash
find . -type file -name '*_*s.tsv' -exec gawk -i inplace '! /"/' {} \;
```

### Create & populate database

Construct a sqlite database.

```bash
sqlite3 -batch booru-chars.db <<'SQL'
.mode tabs
create table files (
  BOORU text not null,
  FID integer not null,
  FILE_NAME text not null,
  TORR_MD5 text,
  ORIG_EXT text,
  ORIG_MD5 text,
  FILE_SIZE text,
  IMG_SIZE_TORR text,
  JQ text,
  TORR_PATH text,
  TAGS_COPYR text,
  TAGS_CHAR text,
  TAGS_ARTIST text,
  primary key(BOORU, FID)
);
create index idx_FILE_NAME on files (FILE_NAME);
create table tags (
  BOORU text,
  FID integer,
  TAG text,
  TAG_ID integer,
  TAG_CAT integer,
  DANB_FR text,
  FOREIGN KEY (BOORU, FID)
    REFERENCES files (BOORU, FID)
    ON DELETE CASCADE
    ON UPDATE NO ACTION
);
create index idx_TAG on tags (TAG);
create index idx_TAG_CAT on tags (TAG_CAT);
create index idx_DANB_FR on tags (DANB_FR);
.import --skip 1 "Safebooru 2021a/V2021A_files.tsv" files
.import --skip 1 "Safebooru 2021a/V2021A_tags.tsv" tags
.import --skip 1 "Safebooru 2021b/V2021B_files.tsv" files
.import --skip 1 "Safebooru 2021b/V2021B_tags.tsv" tags
.import --skip 1 "Safebooru 2021c/V2021C_files.tsv" files
.import --skip 1 "Safebooru 2021c/V2021C_tags.tsv" tags
.import --skip 1 "Safebooru 2021d/V2021D_files.tsv" files
.import --skip 1 "Safebooru 2021d/V2021D_tags.tsv" tags
.import --skip 1 "Safebooru 2022a/V2022A_files.tsv" files
.import --skip 1 "Safebooru 2022a/V2022A_tags.tsv" tags
.import --skip 1 "Safebooru 2022b/V2022B_files.tsv" files
.import --skip 1 "Safebooru 2022b/V2022B_tags.tsv" tags
SQL
```
