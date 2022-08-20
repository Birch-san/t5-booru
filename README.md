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

[`scripts/make_db.sh`](scripts/make_db.sh).

Inputs:

- [`queries/walfie.sql`](queries/walfie.sql)
- `~/machine-learning/booru-chars/Safebooru 2021a/V2021A_files.tsv`
- `~/machine-learning/booru-chars/Safebooru 2021a/V2021A_tags.tsv`
- `~/machine-learning/booru-chars/Safebooru 2021b/V2021B_files.tsv`
- `~/machine-learning/booru-chars/Safebooru 2021b/V2021B_tags.tsv`
- `~/machine-learning/booru-chars/Safebooru 2021c/V2021C_files.tsv`
- `~/machine-learning/booru-chars/Safebooru 2021c/V2021C_tags.tsv`
- `~/machine-learning/booru-chars/Safebooru 2021d/V2021D_files.tsv`
- `~/machine-learning/booru-chars/Safebooru 2021d/V2021D_tags.tsv`
- `~/machine-learning/booru-chars/Safebooru 2022a/V2022A_files.tsv`
- `~/machine-learning/booru-chars/Safebooru 2022a/V2022A_tags.tsv`
- `~/machine-learning/booru-chars/Safebooru 2022b/V2022B_files.tsv`
- `~/machine-learning/booru-chars/Safebooru 2022b/V2022B_tags.tsv`

Outputs:

- `booru-chars.db`

We use `! /\"/` and `grep -v -e "\""` to avoid ingesting double-quote characters. for example, `V2022B_tags.tsv` contains a line with double-quotes:

```
www.zerochan.net	3633380	"I'm_One_Dapping_Fella"_S...	0	3	COPYRIGHT
```

We use `uniq` on `V2021C_tags.tsv` to eliminate duplicate data for `(booru='chan.sankakucomplex.com', FID=25521675)`:

```
chan.sankakucomplex.com	25521675	genshin_impact	1528497	3	genshin_impact
chan.sankakucomplex.com	25521675	genshin_impact	1528497	3	genshin_impact
chan.sankakucomplex.com	25521675	zhongli_(genshin_impact)	1600114	4	genshin_impact
chan.sankakucomplex.com	25521675	zhongli_(genshin_impact)	1600114	4	genshin_impact
```

which would violate the primary key of `tags`. Fortunately the duplicates are contiguous.

### Create tag lists

[`scripts/make_tag_lists.sh`](scripts/make_tag_lists.sh)

Inputs:

- `booru-chars.db`

Outputs:

- `out/general_label_prevalence.raw.tsv`
- `out/name_tokens_prevalence.raw.tsv`

### Create token lists

#### General tokens

[`scripts/split.sh`](scripts/split.sh)

Inputs:

- `out/general_label_prevalence.raw.tsv`

Outputs:

- `out/general_tokens.tsv`

#### Non-general tokens

[`scripts/top.sh`](scripts/top.sh)

Inputs:

- `out/name_tokens_prevalence.raw.tsv`

Outputs:

- `out/label_tokens.tsv`