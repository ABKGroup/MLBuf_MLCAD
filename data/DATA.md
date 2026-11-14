# MLBuf v2 training data (NanGate45)

Version: v2.0 (2026-11). Companion of `ABKGroup/MLBuf_MLCAD` branch `v2` and the pre-trained model
`results/model_dict/mlbuf-v2.pt`. Download: [Google Drive folder](https://drive.google.com/drive/folders/1uprj0g7dzjtwjMCDRIP1jfrt0S7VGP9j?usp=sharing)
(11 archives).

## What it is

rsz buffered-tree dumps written by OpenROAD's timing-driven global placement (RePlAce + virtual repair) for six designs on NanGate45. 

| file | content | used for |
|---|---|---|
| `buffered_trees_pre_<trigger>.csv`  | every net the resizer visited: driver + sinks with OpenSTA values before any buffering | model input; negatives = nets absent from `post` |
| `buffered_trees_post_<trigger>.csv` | the final buffered tree of every net that received >= 1 buffer | labels (buffer type, location, tree) |


## Using it

```bash
# download, verify, extract
sha256sum -c CHECKSUMS.sha256
for f in *.tar.gz; do tar -xzf "$f"; done          # -> <source>/buffered_trees_{pre,post}_*.csv
sha256sum -c MANIFEST.sha256
export MLBUF_TRAINING_DATA=$PWD
python train.py configs/mlbuf_v2_full.yaml --gpu 0     
```

