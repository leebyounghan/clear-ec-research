#!/usr/bin/env python3
"""Rebuild a tier-p3 (or p3b) model archive from the tier-p archive plus the new weight file(s), and check its contents against the tree the build box
tested (deploy lane, 2026-11-14, D-COORD-P3-PREP). Python 3.8+ standard library only (macOS /usr/bin/python3 works); no extraction to disk.

  rebuild   python3 rebuild_model_archive_20261114.py OLD.tar.gz PATCH_DIR NEW.tar.gz
              PATCH_DIR = the folder downloaded from the box (upload_20261114/<variant>_model_patch): model_swap.tsv, model_tree_sha256_<variant>.txt
              (per-file sha256 of the model tree the box container test mounted) and files/<new path>.
              1 sha256 of OLD must be the tier-p archive's (fff3ec09...); 2 each new file's sha256 must match model_swap.tsv;
              3 one streaming pass OLD -> NEW.tar.gz.part: every member copied unchanged except the old paths of model_swap.tsv, which are left out;
                the new files are added (mode 0644, owner 0:0, missing parent directories 0755); gzip level 6 like pigz's default;
              4 an independent second pass reads NEW.tar.gz.part back and lists every file (sha256, size, path), checking that it holds only regular
                files and directories, that every file is world-readable and every directory world-traversable (the container runs as uid 10001),
                no duplicate names and no macOS '._' files; the list must equal model_tree_sha256_<variant>.txt line for line;
              5 only then NEW.tar.gz.part is renamed to NEW.tar.gz, and NEW.tar.gz.sha256 holds the archive's own sha256 (for the upload record).
              Refuses an existing NEW (or NEW.part). Exit 0 = contents equal to the box-tested tree.
  list-dir  python3 rebuild_model_archive_20261114.py --list-dir DIR [OUT]      (the box: per-file list of the extracted tree it tests)
  list-tar  python3 rebuild_model_archive_20261114.py --list-tar ARCHIVE [OUT]  (any .tar.gz: per-file list of its contents)
List format: '<sha256>  <bytes>  <path>' sorted by path bytes, path relative to the archive root without './'."""
import sys, os, gzip, tarfile, hashlib, time, stat

TIER_P_SHA256 = 'fff3ec09e155841f43057316e8398fe01ef57c5f18b97530a5daa94df43f58c5'
CHUNK = 1 << 22


def norm(name):
    while name.startswith('./'): name = name[2:]
    return name.rstrip('/')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''): h.update(b)
    return h.hexdigest()


def write_list(rows, out):
    txt = ''.join(f'{h}  {n}  {p}\n' for p, (h, n) in sorted(rows.items(), key=lambda kv: kv[0].encode()))
    if out: open(out, 'w').write(txt)
    else: sys.stdout.write(txt)
    return txt


def list_dir(d):
    rows, bad = {}, []
    for dp, dn, fn in os.walk(d):
        for x in dn:
            st = os.lstat(os.path.join(dp, x))
            if stat.S_ISLNK(st.st_mode): bad.append('symlink dir ' + os.path.join(dp, x))
            elif (st.st_mode & 0o005) != 0o005: bad.append('dir not world-traversable ' + os.path.join(dp, x))
        for x in fn:
            p = os.path.join(dp, x); st = os.lstat(p)
            if not stat.S_ISREG(st.st_mode): bad.append('not a regular file ' + p); continue
            if not st.st_mode & 0o004: bad.append('file not world-readable ' + p)
            rows[os.path.relpath(p, d)] = (sha256_file(p), st.st_size)
    return rows, bad


def list_tar(path):
    rows, bad = {}, []
    with gzip.open(path, 'rb') as g, tarfile.open(fileobj=g, mode='r|') as t:
        for m in t:
            n = norm(m.name)
            if n in ('', '.'): continue
            if os.path.basename(n).startswith('._'): bad.append('macOS metadata file ' + n)
            if m.isdir():
                if (m.mode & 0o005) != 0o005: bad.append('dir not world-traversable ' + n)
                continue
            if not m.isreg(): bad.append(f'not a regular file (type {m.type!r}) {n}'); continue
            if not m.mode & 0o004: bad.append('file not world-readable ' + n)
            if n in rows: bad.append('duplicate ' + n)
            h = hashlib.sha256(); f = t.extractfile(m); k = 0
            for b in iter(lambda: f.read(CHUNK), b''): h.update(b); k += len(b)
            if k != m.size: bad.append('short member ' + n)
            rows[n] = (h.hexdigest(), m.size)
    return rows, bad


def read_list(p):
    out = {}
    for line in open(p):
        if not line.strip(): continue
        h, n, rel = line.rstrip('\n').split('  ', 2); out[rel] = (h, int(n))
    return out


class HashWriter:
    def __init__(self, f): self.f, self.h, self.n = f, hashlib.sha256(), 0
    def write(self, b): self.h.update(b); self.n += len(b); return self.f.write(b)
    def flush(self): self.f.flush()
    def close(self): self.f.close()


class HashReader:
    def __init__(self, f): self.f, self.h = f, hashlib.sha256()
    def read(self, n=-1):
        b = self.f.read(n); self.h.update(b); return b


def rebuild(old, patch, new):
    t0 = time.time(); part = new + '.part'
    for p in (new, part):
        if os.path.exists(p): sys.exit(f'refusing: {p} exists')
    swap = [l.rstrip('\n').split('\t') for l in open(os.path.join(patch, 'model_swap.tsv')) if l.strip() and not l.startswith('#')]
    lists = [f for f in os.listdir(patch) if f.startswith('model_tree_sha256_') and f.endswith('.txt')]
    if len(lists) != 1: sys.exit(f'expected one model_tree_sha256_<variant>.txt in {patch}, found {lists}')
    box = read_list(os.path.join(patch, lists[0]))
    print(f'[1/5] sha256 of {old} (tier-p archive) ...', flush=True)
    h = sha256_file(old)
    if h != TIER_P_SHA256: sys.exit(f'old archive sha256 {h} is not the tier-p archive {TIER_P_SHA256}')
    print(f'      ok {h[:16]} ({time.time() - t0:.0f} s)', flush=True)
    rm = {norm(o) for o, n, s in swap}; add = {}
    for o, n, s in swap:
        f = os.path.join(patch, 'files', norm(n))
        if sha256_file(f) != s: sys.exit(f'[2/5] new file {f} does not match model_swap.tsv')
        add[norm(n)] = f
    print(f'[2/5] {len(add)} new file(s) match model_swap.tsv; leaving out {len(rm)} old path(s)', flush=True)
    print(f'[3/5] writing {part} (one streaming pass, gzip level 6) ...', flush=True)
    seen_rm, dirs, prefix, nfile, nbytes = set(), set(), None, 0, 0
    with open(part, 'xb') as raw:
        hw = HashWriter(raw)
        with gzip.GzipFile(filename='', mode='wb', fileobj=hw, compresslevel=6, mtime=0) as gz, tarfile.open(fileobj=gz, mode='w|', format=tarfile.GNU_FORMAT) as out, \
                gzip.open(old, 'rb') as g, tarfile.open(fileobj=g, mode='r|') as inp:
            for m in inp:
                n = norm(m.name)
                if prefix is None and n not in ('', '.'): prefix = './' if m.name.startswith('./') else ''
                if m.isdir(): dirs.add(n)
                if m.isreg() and n in rm: seen_rm.add(n); continue
                if n in add: sys.exit(f'new path {n} is already in the old archive')
                out.addfile(m, inp.extractfile(m) if m.isreg() else None)
                if m.isreg(): nfile += 1; nbytes += m.size
                if nfile and nfile % 50 == 0 and m.isreg(): print(f'      {nfile} files, {nbytes / 1e9:.1f} GB ({time.time() - t0:.0f} s)', flush=True)
            if seen_rm != rm: sys.exit(f'old paths not found in the old archive: {sorted(rm - seen_rm)}')
            prefix = prefix or ''
            for n, f in sorted(add.items()):
                parts = n.split('/')
                for i in range(1, len(parts)):
                    dd = '/'.join(parts[:i])
                    if dd not in dirs:
                        ti = tarfile.TarInfo(prefix + dd); ti.type = tarfile.DIRTYPE; ti.mode = 0o755; ti.mtime = int(time.time())
                        ti.uid = ti.gid = 0; ti.uname = ti.gname = ''; out.addfile(ti); dirs.add(dd)
                st = os.stat(f); ti = tarfile.TarInfo(prefix + n); ti.size = st.st_size; ti.mode = 0o644; ti.mtime = int(st.st_mtime)
                ti.uid = ti.gid = 0; ti.uname = ti.gname = ''
                with open(f, 'rb') as fh: out.addfile(ti, fh)
                nfile += 1; nbytes += st.st_size
    print(f'      {nfile} files, {nbytes / 1e9:.2f} GB written, archive {hw.n / 1e9:.2f} GB ({time.time() - t0:.0f} s)', flush=True)
    print(f'[4/5] reading {part} back and comparing with {lists[0]} ...', flush=True)
    rows, bad = list_tar(part)
    write_list(rows, part + '.files.txt')
    miss = sorted(set(box) - set(rows)); extra = sorted(set(rows) - set(box)); diff = sorted(k for k in set(box) & set(rows) if box[k] != rows[k])
    print(f'      archive: {len(rows)} files {sum(v[1] for v in rows.values()):,} bytes; box tree: {len(box)} files {sum(v[1] for v in box.values()):,} bytes', flush=True)
    if bad or miss or extra or diff:
        for x in bad[:10]: print('      PROBLEM', x)
        for k in miss[:10]: print('      MISSING from the new archive', k)
        for k in extra[:10]: print('      EXTRA in the new archive', k)
        for k in diff[:10]: print('      DIFFERENT', k, box[k], rows[k])
        sys.exit(f'CONTENTS DO NOT MATCH the box-tested tree; {part} kept for inspection, not renamed')
    os.rename(part, new); os.rename(part + '.files.txt', new + '.files.txt')
    open(new + '.sha256', 'w').write(f'{hw.h.hexdigest()}  {os.path.basename(new)}\n')
    print(f'[5/5] CONTENTS MATCH the box-tested tree ({len(rows)} files). {new}  {os.path.getsize(new):,} bytes  sha256 {hw.h.hexdigest()}  '
          f'({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] in ('--list-dir', '--list-tar'):
        rows, bad = (list_dir if a[0] == '--list-dir' else list_tar)(a[1])
        write_list(rows, a[2] if len(a) > 2 else None)
        sys.stderr.write(f'{len(rows)} files, {sum(v[1] for v in rows.values()):,} bytes; problems: {len(bad)} {bad[:5]}\n')
        sys.exit(1 if bad else 0)
    if len(a) != 3: sys.exit(__doc__)
    rebuild(*a)
