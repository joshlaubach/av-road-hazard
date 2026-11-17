# REPORT-F: reproducibility records (branch fix/reproducibility-records, commit 02d3fcf)

## Committed under eval/ids/
| File | Size | SHA-256 | Matches record? |
|---|---|---|---|
| extra_negatives_batch3.json | 6,483 B, 68 entries | 5269526479c65963c3e21f2615497bf15fe65b65c2732740ddb6640181e823b8 | Yes: `extra_negatives_sha256` in best.json of tile-fold0-hn1 (repeat 20) and tile-fold0-hn2 (repeat 10) |
| probe_negatives_batch3.json | 2,187 B, 23 entries | 0d83b44a245778b69c71ffc058a9b0f42a2874be1471109be2af82eee09422f6 | Yes: `provenance.classifier.files.probe` in eval/results.json (the README's 23 held-out clean tiles) |
| .gitattributes | `*.json -text` | n/a | Keeps CRLF and no final newline, so hashes match in git |
| README.md | short | n/a | Each file's source, SHA-256 and run |

Hashes were checked on the committed blobs (`git show HEAD:...`). A content scan of all 8 source files found no base64, data:image, absolute paths, emails, owner name or "nuscenes". The committed files hold only `{image_id, row, col}` RDD2022 IDs. Runs hn1 and hn2 used the same hard-negative file.

## Owner decisions
- **splits.json not committed.** 7,905,876 B, SHA ac18b6b881c9c9a187066243ac609be2c376c1168bea3dc845eec9d5a5b6e4b6 (matches `split_file_sha256`). IDs only, but `.pre-commit-config.yaml` sets `check-added-large-files --maxkb=1024`. Options: exclude this path from the limit, store it gzipped with the raw SHA-256 recorded, or use Git LFS. It is CRLF too, so it needs the `-text` rule.
- **Pre-commit risk.** `end-of-file-fixer` and `trailing-whitespace` would rewrite the eval/ids JSON files and break their hashes if someone runs `pre-commit run --all-files`. Suggest adding `exclude: ^eval/ids/.*\.json$` to both hooks (outside this task's files).
- **Inspected, not copied:** splits.manifest.json (6.48 MB; IDs, per-image SHA-256, labels), groups.json (3.36 MB; ID to group), sample_index.json (6.35 MB; IDs, labels, relative paths), tile_index.json (3.92 MB; IDs, sizes, label boxes, tile settings; SHA matches `tile_index_sha256`), exclusions.json (328 KB; IDs, reasons, counts; under the size limit, could be committed).

## Tests
pytest: 1200 passed, 4 skipped, 1 deselected (baseline). check-added-large-files, check-json and detect-private-key passed on the new files. No evaluation run; eval/results.json unchanged.
