# Release process

A release is a tagged, citable snapshot of the study. The steps below keep the
version, the changelog and the citation metadata in agreement — CI fails if the
version in `CITATION.cff` and `.zenodo.json` disagree.

## 1. Prepare

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run mypy src
poetry run pytest
poetry build
```

## 2. Bump the version in five places

The version appears in:

| File             | Field                         |
| ---------------- | ----------------------------- |
| `pyproject.toml` | `[project] version`           |
| `CITATION.cff`   | `version` and `date-released` |
| `.zenodo.json`   | `version`                     |
| `CHANGELOG.md`   | a new section heading         |
| `README.md`      | `version` in the BibTeX block |

CI checks the first three against each other. The changelog heading and the
BibTeX block are on you.

This project uses semantic versioning, read for a research codebase as:

- **major** — the experimental protocol changed in a way that makes previously
  reported costs incomparable (different splits, different cost constants);
- **minor** — new model families, studies or metrics;
- **patch** — fixes and tooling that leave reported results unchanged.

If a change moves a published number, say so explicitly in the changelog entry.

## 3. Tag and release

```bash
git commit -am "Release 0.3.0"
git tag -a v0.3.0 -m "Release 0.3.0"
git push origin main --tags
gh release create v0.3.0 --title "0.3.0" --notes-file <(sed -n '/## 0.3.0/,/## 0.2.0/p' CHANGELOG.md)
```

## 4. Archive on Zenodo

`.zenodo.json` in the repository root supplies the deposit metadata: title,
description, license, keywords, ORCID and the link to the source dataset. It is
read automatically when Zenodo archives a GitHub release.

**The GitHub integration requires a public repository.** While this repository
is private, use the manual route.

### If the repository is public

1. Sign in at [zenodo.org](https://zenodo.org) with GitHub and grant the
   `admin:repo_hook` scope.
2. Open **Zenodo → GitHub**, find `scania-aps-cost` and switch it **on**.
3. Publish a GitHub release. Zenodo archives the tarball, reads
   `.zenodo.json`, and mints a DOI within a few minutes.
4. Copy the **concept DOI** — the one that always resolves to the newest
   version — and add its badge under the title in `README.md`:

   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

5. Add the DOI to `CITATION.cff` as a top-level `doi:` field so the citation
   metadata and the archive agree.

Only releases published *after* the switch is enabled are archived; enabling it
does not backfill earlier tags.

### While the repository is private

1. Build the artifact: `poetry build`, or `git archive --format=tar.gz -o scania-aps-cost-0.3.0.tar.gz v0.3.0`.
2. Start a new upload at [zenodo.org/uploads/new](https://zenodo.org/uploads/new).
3. Fill the form from `.zenodo.json` — it is the source of truth for the
   metadata, even when it is not being read automatically.
4. Set access to whatever the work requires; a **restricted** or **embargoed**
   record still receives a DOI, so the work stays citable without publishing
   the source.
5. Record the resulting DOI in `CITATION.cff` and `README.md` as above.

## 5. Verify

- The GitHub release page shows the right notes and artifacts.
- The Zenodo record shows the correct author, ORCID and license.
- `CITATION.cff` renders in the sidebar under "Cite this repository".
