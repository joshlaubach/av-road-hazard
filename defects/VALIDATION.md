# Annotation validation

Before preparing splits on the PC, export a validated sample index:

```powershell
python -m defects.validate_annotations "$env:RDD2022_ROOT" --report artifacts/validation.json --sample-index artifacts/samples.json
```

Set `sample_index` in the training YAML to the generated index (relative to the
config file, as with existing training commands). Existing indexes are not
automatically revalidated: regenerate the index before making new splits.
Do not reuse existing splits or a dataset identity after changing sample membership.

The command reads annotations and verifies image files; it does not download data,
train a model, or change source annotations. Keep outputs local, outside git.
Image resolution follows the same layout search as `defects.voc.find_image`.

Malformed XML, missing boxes/class codes, missing/nonfinite/noninteger coordinates,
nonpositive boxes, out-of-image boxes, unreadable images, and conflicting image
dimensions exclude the entire image. Missing XML size metadata is allowed because
actual image dimensions supply the bounds. Whole-image exclusion prevents an
omitted defect from becoming background; tile-level salvage is not implemented.

`validation.json` records included/excluded image counts, unexpected-code counts,
and dataset-relative sample ids with exclusion reasons. Reasons are deduplicated
per image, not per-box counts. `samples.json` contains only eligible samples with
four binary class-code labels and image paths relative to that index.

Valid empty annotations remain negative samples. The existing scope policy stays:
out-of-scope-only images are excluded, while structurally valid mixed-code images
retain in-scope labels. Unexpected codes are reported; their meaning is not inferred.
Structural validity does not establish semantic label correctness. Human-readable
class names remain unverified, and semantic labels are never automatically corrected.

Change note: `build_samples()` now validates structure and image integrity before
returning training samples. Its report retains the existing unexpected-code counter
and `excluded_ids`, adding `exclusion_reasons` and `write_json(path)`.
