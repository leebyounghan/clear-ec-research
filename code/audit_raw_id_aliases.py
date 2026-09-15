#!/usr/bin/env python3
"""Recover separator-only raw/CSV ID aliases without changing active OOF rows."""
import argparse
import csv
import json
import re
from pathlib import Path


def canonical(value):
    return re.sub(r'[\s-]', '', value.removesuffix('.mha').removesuffix('.bmp'))


def unique_index(values):
    out = {}
    for value in values:
        key = canonical(value)
        if key in out:
            raise ValueError(f'Ambiguous ID normalization: {out[key]!r}, {value!r}')
        out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('datasets/clear_ec_fullframe_20260904'))
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root/'fullframe_manifest.json').read_text())
    with (root/'source_metadata/final_train_ids.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    raw_ids = {p.stem for p in (root/'train_mha').glob('*.mha')}
    label_ids = {r['ID'] for r in rows}
    assert len(rows) == len(raw_ids) == len(label_ids) == manifest['raw_file_count'] == 9000
    # Preserve exact matches first: distinct existing images can share a normalized
    # form, so normalizing all names would incorrectly merge them.
    exact = raw_ids & label_ids
    raw = unique_index(raw_ids - exact)
    labels = unique_index(label_ids - exact)
    assert set(raw) == set(labels), 'Unmatched IDs do not give a one-to-one match'
    assert not set(raw).intersection(canonical(i) for i in exact), 'Alias also matches an existing image'
    aliases = []
    corrected = []
    for row in rows:
        raw_id = row['ID'] if row['ID'] in exact else raw[canonical(row['ID'])]
        assert row['slide_id'].replace('-', '') in canonical(raw_id)
        fixed = dict(row, ID=raw_id, source_csv_id=row['ID'])
        corrected.append(fixed)
        if raw_id != row['ID']:
            aliases.append(dict(raw_id=raw_id, source_csv_id=row['ID'], slide_id=row['slide_id']))
    assert {r['raw_id'] for r in aliases} == set(manifest['extra_raw_ids'])
    report = dict(archive_sha256=manifest['archive_sha256'], raw_count=len(raw_ids),
        label_count=len(rows), aliases=aliases, normalized_bijection=True,
        rule='Preserve exact matches first. Remove whitespace/hyphens only for unmatched IDs; require unique keys and no collision with exact matches. No fuzzy matching.',
        scope='All 9000 labels recovered for future retraining. Existing 8954 OOF rows, folds, caches and jobs remain unchanged.')
    for dest, content in [(root/'raw_id_alias_audit.json', json.dumps(report,indent=2))]:
        if dest.exists():
            assert dest.read_text() == content
        else:
            temp=dest.with_suffix('.tmp'); temp.write_text(content); temp.rename(dest)
    dest=root/'source_metadata/final_train_ids_9000_reconciled.csv'
    if not dest.exists():
        temp=dest.with_suffix('.tmp')
        with temp.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields+['source_csv_id'])
            writer.writeheader(); writer.writerows(corrected)
        temp.rename(dest)
    else:
        with dest.open(newline='') as stream: assert list(csv.DictReader(stream)) == corrected
    print(json.dumps(dict(raw_count=len(raw_ids),label_count=len(rows),recovered_aliases=len(aliases),
        existing_oof_unchanged=True)))


if __name__=='__main__': main()
