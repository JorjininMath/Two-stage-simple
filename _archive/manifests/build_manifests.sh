#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"

files_tmp="$(mktemp)"
hash_paths_tmp="$(mktemp)"
hashes_tmp="$(mktemp)"
trap 'rm -f "${files_tmp}" "${hash_paths_tmp}" "${hashes_tmp}"' EXIT

printf 'path\tbytes\tmodified_at\n' > "${files_tmp}"
find _archive -type f \
  ! -path '_archive/manifests/FILES.tsv' \
  ! -path '_archive/manifests/SHA256SUMS' \
  -exec stat -f $'%N\t%z\t%Sm' -t '%Y-%m-%dT%H:%M:%S%z' {} \; \
  | LC_ALL=C sort >> "${files_tmp}"
mv "${files_tmp}" _archive/manifests/FILES.tsv

# Hash human-authored source/control files and compact summaries. Large raw
# output, cache, data, and nested Git trees remain represented by FILES.tsv.
find _archive -type f \
  ! -path '*/.git/*' \
  ! -path '*/output/*' \
  ! -path '*/output_*/*' \
  ! -path '*/data/*' \
  ! -path '_archive/manifests/SHA256SUMS' \
  -print > "${hash_paths_tmp}"
find _archive -type f \
  ! -path '*/.git/*' \
  \( -iname '*summary*.csv' -o -iname '*summary*.txt' \) \
  -size -10M -print >> "${hash_paths_tmp}"

LC_ALL=C sort -u "${hash_paths_tmp}" | while IFS= read -r archive_file; do
  shasum -a 256 "${archive_file}"
done > "${hashes_tmp}"
mv "${hashes_tmp}" _archive/manifests/SHA256SUMS

printf 'Wrote %s file records and %s checksums.\n' \
  "$(( $(wc -l < _archive/manifests/FILES.tsv) - 1 ))" \
  "$(wc -l < _archive/manifests/SHA256SUMS)"
