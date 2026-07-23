# Paper Snapshots

This folder stores PDF snapshots intended for sharing or historical reference.

- `current/` contains at most one approved, unversioned shareable snapshot.
- `archive/YYYY-MM/` keeps superseded snapshots grouped by sharing date.

Current shareable snapshot:

- `current/CKME_Adaptive_H.pdf` -- advisor-ready, unversioned snapshot built
  on 2026-07-23 from
  `manuscript/journal_scale_adaptive/scale_adaptive_ckme_cp.tex` at source
  commit `fe5922013d4cd7f434e071b0b69ee944d94d779e`. The 27-page PDF passed
  LaTeX log checks and rendered-page visual review. Its SHA-256 is
  `b2ae7fad87aa39599be17d2238fe65e72264a3f072b11e92757cae3d857888c7`.

Archived snapshots:

- `archive/2026-06/CKME_CP_20260630.pdf` — historical cleaned snapshot with numerical evidence
  temporarily hidden. It predates the current estimator and calibration
  protocol and should not be used as the latest method description. It was
  compiled on 2026-06-30 from the source then named
  `target_aware_scale_adaptive_ckme_cp.tex`; that active source was renamed
  `scale_adaptive_ckme_cp.tex` on 2026-07-23.
- `archive/2026-06/CKME_CP_20260616_v0.1.pdf` — original June 16 snapshot compiled from
  the source then named `target_aware_scale_adaptive_ckme_cp.tex` on
  2026-06-16.

Do not put advisor-returned editable files here. Store them in a private dated
round under `manuscript/advisor_feedback/rounds/`; keep the exact sent PDF path
in that round's manifest.
