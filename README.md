# t5-booru

Trying here to fine-tune an off-the-shelf T5 checkpoint on tag sets from Danbooru images, to understand (for example) that `usada_pekora` implies `bunny_ears` and `carrot`. The hope is also that knowledge T5 learned in pre-training (from the C4 corpus) can hydrate our Danbooru tags with more implications (e.g. that they're human).

## Setup repo

```bash
python3 -m venv venv
. ./venv/bin/activate
pip install --upgrade pip
pip install transformers
pip install --pre "torch>1.13.0.dev20220610" --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# for Jupyter notebooks
pip install ipython jupyter ipywidgets
```

## Make a database of Danbooru tags, from the booru-chars distribution

Download a recent booru-chars distribution, e.g. https://nyaa.si/view/1486179.  
You can download with a bittorrent client such as [qBittorrent](https://www.qbittorrent.org/download.php).  
Specifically download from it the tags file, `V2022A_tags.tsv` (124.2 MiB).

Construct from it a sqlite database, like this?

```bash
sqlite3 -batch booru-chars.db <<'SQL'
.mode tabs
.import "V2022A_tags.tsv" tags
SQL
```