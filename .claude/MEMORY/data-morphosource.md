# MorphoSource Data Acquisition

`scripts/download_morphosource_project.py` pulls openly-downloadable media for a MorphoSource
project straight into `data/`. It defaults to project **000381689** — *"UF Photogrammetry scans
at the Florida Museum of Natural History"* — the upstream source of the UF scenes that
`pipeline/preparation/prepare_uf_dataset.py` consumes.

## Install

The script needs the `morphosource` package (pinned as `morphosource>=1.2.0` in
`requirements.txt`, so `scripts/setup_common.sh` step 2/7 installs it). For an existing env:

```bash
python -m pip install --upgrade morphosource
```

Pure Python (deps `pygbif`, `requests`), no numpy dependency, so the repo's `numpy<2` pin is
unaffected.

## API key

Export `MORPHOSOURCE_API_KEY` (MorphoSource → Dashboard → Profile → *View API Key*). Only
downloads need it; `--dry-run` works without. Downloading implies consent to the
[MorphoSource user agreements](https://www.morphosource.org/terms) for those files.

## What is in project 000381689

Every open medium is one of three kinds, one image series plus one or two meshes per specimen:

| Kind | File | Open count | Total |
|------|------|-----------:|------:|
| `images` | `<specimen>_images.zip` — `cameraN/*.jpg` + `*.jpg.mask.png` | 662 | 737 GB (avg 1.1 GB, max 14 GB) |
| `highpoly` | `<specimen>_mesh_highpoly.zip` | 478 | ~71 GB |
| `lowpoly` | `<specimen>_mesh_lowpoly[_edited].zip` | 482 | ~60 GB |
| **all** | | **1,818** (596 specimens) | **~869 GB** |

Because the whole project is far too big to mirror, the default is a **seeded random sample of
3 specimens**, image series + high poly mesh — roughly 3.2 GB.

## Flags

```
--project-id 000381689        MorphoSource project id
--output-dir PATH             default data/morphosource/<project_id>
--kinds images highpoly       subset of {images, lowpoly, highpoly}; default "images highpoly"
--num-specimens 3             size of the seeded sample
--seed 0                      sample seed
--all-specimens               every eligible specimen instead of a sample
--specimen ID [ID ...]        explicit physical object ids; bypasses sampling
--dry-run                     write manifest + print the size table, download nothing
--extract                     unzip each bundle beside itself
--overwrite                   re-download even when a complete file is present
--use-statement TEXT          sent with each download request (default: research statement)
--use-categories Research     MorphoSource use categories
```

A specimen is *eligible* only if it has at least one open medium of **every** requested kind
(422 for the default `images highpoly`). `--specimen` skips that check and warns about gaps.

## Output layout

```
data/morphosource/000381689/
  manifest.json                        # project, selection args (incl. seed), every planned file
  UF_Fish_233247__000811670/           # sanitised physical_object_title + "__" + object id
    metadata.json                      # this specimen's slice of the manifest
    UF_Fish_233247_images.zip
    UF_Fish_233247_mesh_highpoly.zip
    UF_Fish_233247_images/             # only with --extract
      camera1/camera1_IMG_8717.JPG
      camera1/camera1_IMG_8717.jpg.mask.png
      ms_usage_std_comm_no_rearc_ms_3d_limited.pdf
      media-manifest-<uuid>.csv / .xlsx
```

**A download is a bundle, not the media file.** Each zip contains
`Media <id> - <title>/<name>-<id>.zip` — the actual payload — plus the usage agreement PDF and
the media manifests. `--extract` unwraps both layers and deletes the inner zip, so the payload
lands directly under `<name>_images/` (a plain `unzip` leaves you one level short, holding a
gigabyte-sized inner zip).

Meshes unwrap to `object_full.obj` + `.mtl` + `object_full_diffuse.1001.png`; image series
unwrap to `cameraN/` dirs — exactly the mixed image+mask layout `prepare_uf_dataset.py` parses
(`*.JPG` beside `*.jpg.mask.png`, including the doubled `.jpg.mask.jpg.mask.png` suffix it
strips). `prepare_uf_dataset.py` reads a **flat** directory, so feed it one camera at a time:

```bash
python scripts/download_morphosource_project.py --num-specimens 1 --extract
python pipeline/preparation/prepare_uf_dataset.py \
    data/morphosource/000381689/UF_Fish_181080__000816964/UF_Fish_181080_images/camera1 \
    --out data/uf_fish_181080_cam1
```

Downloads are atomic (`.zip.part` → `os.replace`) and validated with `zipfile.is_zipfile`, so an
interrupted pull resumes by simply re-running: a re-run skips any file that already reads as a
complete zip. Note it cannot skip on size — MorphoSource re-packs each bundle, so what lands on
disk never matches the `file_size_all` the API advertises. Extraction rejects zip members with
absolute or `..` paths.

## MorphoSource API gotchas

These drove the implementation and are not obvious from the package README:

- **`search_media()` has no project filter.** Its only params are
  `query/media_type/taxonomy_gbif/visibility/media_tag/per_page/page`. The REST API *does* have
  a `project` facet keyed by project id, so the script calls the package's internal
  `morphosource.fetch.fetch_items(url=Endpoints.MEDIA, params={"f.project": ...})` and wraps the
  raw dicts in `morphosource.search.Media`. That reuses paging, `download_bundle()` and
  `get_file_metadata()` — hence the `>=1.2.0` pin, since these are internal names.
- **Every API field is wrapped in a list** (`{"id": ["000811670"], ...}`); the `first()` helper
  unwraps them.
- **`media.visibility` is the literal `"open"`**, while the facet value is `"Open Download"`
  (`DownloadVisibility.OPEN`). The script filters server-side then re-asserts client-side.
- **`file_name` is empty in search results** for 1,803 of the 1,818 open media. Real names come
  from `GET /api/media/<id>/file-metadata`, one request each.
- **`short_description` is the cheap kind signal**: `"High polygon mesh"` /
  `"Low polygon mesh"` / `"Edited low polygon mesh"`. It is absent on 196 open meshes, which
  fall back to the file name, then to size-ranking the two sibling meshes under one
  `media_parent_id`.
- **Kind classification for *sampling* is deliberately network-free.** `/file-metadata` 503s
  under bursts, so a classification that depended on it made the same `--seed` pick different
  specimens run to run. `classify_media()` uses only the search response; `refine_kinds()` spends
  requests afterwards, on the selected specimens only.
- Those `/file-metadata` lookups are rate-limited: the script uses 2 workers with exponential
  backoff (`METADATA_WORKERS` / `METADATA_RETRIES` / `METADATA_BACKOFF`).
- **`www.morphosource.org` HTML is behind Cloudflare** (browser fetches get *Access Denied*),
  but `/api/...` is open. Project metadata comes from `GET /api/projects/<id>` →
  `response.collection.title[0]`; the package has no helper for it.
- `DownloadConfig` raises `ValueError` unless `use_statement` plus one of
  `use_categories` / `use_category_other` is given.
