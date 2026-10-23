#!/usr/bin/env bash
# Cut a release: one tag that names the commit the hub builds every provider download from
# (install.sh, bmb-eval.tar.gz, sandbox-image.tar).
#   deploy/release.sh v0.2.0           # bump the version, commit, tag; prints the push command
#   deploy/release.sh v0.2.0 --push    # ...and push main and the tag
# Then `deploy/gcp/hub deploy v0.2.0` puts that exact tag on the hub, and
# https://<hub>/release shows the tag with the sha256 of all three downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
V="${1:?usage: deploy/release.sh vX.Y.Z [--push]}"
[[ "$V" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "the version must look like v1.2.3" >&2; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || { echo "release from main" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "commit or stash your changes first" >&2; exit 1; }
git fetch -q --tags origin
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] ||
  { echo "main isn't in sync with origin/main: pull or push first" >&2; exit 1; }
! git rev-parse -q --verify "refs/tags/$V" >/dev/null || { echo "$V already exists" >&2; exit 1; }
command -v uv >/dev/null || { echo "needs uv (https://astral.sh/uv)" >&2; exit 1; }

# The lock must still cover requirements.txt, and the deploy script must stamp the same files.
uv run -q --all-extras pytest -q tests/test_release.py
# pyproject.toml and uv.lock carry the same number as the tag.
uv version -q --no-sync "${V#v}"
git commit -qam "Release $V"
git tag -a "$V" -m "bmb-eval $V"
echo "tagged $V at $(git rev-parse --short HEAD)"
if [ "${2:-}" = --push ]; then
  git push -q origin main "$V"
  echo "pushed. Deploy it with:  deploy/gcp/hub deploy $V"
else
  echo "push with:    git push origin main $V"
  echo "then deploy:  deploy/gcp/hub deploy $V"
fi
