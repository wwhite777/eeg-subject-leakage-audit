# Transformation record — public copy of REVISION_PLAN_v1.md

The analysis plan of the first revision was frozen on 2026-09-16 and its SHA-256 recorded before any of the first-revision runs:

    8d9b1f5b55151c07508fcf9d0dd4aead1723c0c30e449b79153dcc9abdf05b73  research/REVISION_PLAN_v1.md   (frozen bytes, private copy)

The copy released in the public repository (tag `revision-v2.1` and later, path `research/REVISION_PLAN_v1.md`) has a different checksum:

    334cf5c0f24ce07ddb339d9b5749ff50edded018050b8425514838eb8ac64588  research/REVISION_PLAN_v1.md   (distributed bytes)

Reason. The release-staging script (`test/experiments/revision/stage_public_repo.sh`) replaces the journal's name in every released
Markdown, Python and shell file so that the public repository does not name the journal. Writing J for the
journal's three-letter acronym (the journal is named in the article's data-availability statement), the three substitutions, applied in
this order with GNU sed, are:

    s/J-revision/revision/g;  s/J major revision/major revision/g;  s/\bJ\b/the journal/g

They change exactly one line of the plan, its title line ("J major revision of the subject-leakage audit" → "major revision of the
subject-leakage audit"). Nothing else differs: no definition, rule, number or date was altered.

Verification (2026-10-30): applying the three substitutions to the frozen private bytes reproduces the distributed checksum exactly
(sed … REVISION_PLAN_v1.md | sha256sum → 334cf5c0…). Conversely, replacing "major revision of the subject-leakage audit" in the first
line of the distributed file by "J major revision of the subject-leakage audit" (with J spelled out) restores the frozen bytes and the
checksum 8d9b1f5b…, which any reader can check.

Two companion files are released: `research/REVISION_PLAN_v1.sha256` records the frozen (private) checksum 8d9b1f5b… and `research/REVISION_PLAN_v1.distributed.sha256` records the checksum 334cf5c0… of the distributed bytes (added 2026-10-30 after an independent reviewer noted that the single companion file could be read as the distributed checksum). The frozen file records the checksum 8d9b1f5b…, which is why the two did not
match; this record resolves the inconsistency by stating both checksums and the exact transformation. The plan of the second revision
(`research/REVISION_PLAN_v2.md`, frozen 2026-10-30, SHA-256 888c7635ff5d71b55023fc8ccacc19f7ca79cd7097cffe370fdb045d916bf096) was
written without the journal's name and is released byte-for-byte, so its recorded and distributed checksums coincide.
