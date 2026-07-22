.PHONY: test test-mm1 compile check-assets export-assets check-archive check-r career-status help

help:
	@printf '%s\n' \
	  'make test           Core import/path/pipeline tests' \
	  'make test-mm1       M/M/1 feasibility tests (pytest)' \
	  'make compile        Compile Python source without running experiments' \
	  'make check-assets   Verify manuscript asset hashes' \
	  'make export-assets  Refresh the explicit manuscript asset allowlist' \
	  'make check-archive  Verify archived source/control hashes' \
	  'make check-r        Parse R benchmark sources' \
	  'make career-status  Validate Career OS export markers'

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

test-mm1:
	python -m pytest experiments/mm1_feasibility/tests -q

compile:
	python -m compileall -q src experiments tests tools

check-assets:
	python tools/export_manuscript_assets.py --check

export-assets:
	python tools/export_manuscript_assets.py

check-archive:
	shasum -a 256 -c _archive/manifests/SHA256SUMS

check-r:
	Rscript -e "parse(file='benchmarks/dcp/dcp_methods.R')"
	Rscript -e "parse(file='benchmarks/dcp/run_one_case.R')"

career-status:
	python tools/export_project_status.py --check
