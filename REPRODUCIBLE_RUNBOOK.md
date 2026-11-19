# Reproduction

Python 3.12, NumPy 2.4.6 and SciPy 1.17.1 were used. Install from requirements.txt in an isolated environment if needed.

From this repository root:

```
python src/test_r14_contracts.py
python src/reproduce_release.py
python src/release_gate_v2.py --help
```

The first command checks the perfect-reference finite-population implementation. The second reads only `inputs/`, independently enumerates the mixed-placement identity on 240 small configurations, calculates the fallible-answer counterexample, and regenerates `outputs/reproduced_results.json`. It checks the two headline cohort expectations to 1e-10. Runtime depends on the largest four-subsystem designs.

Inputs are derived counts from frozen review trajectories and authored-property records. They reproduce the release calculations, not the upstream IFC extraction, hidden-state simulation, professional adjudication or national-code interpretation. Source URLs and hashes are supplied in DATASETS_AND_LINKS.csv. The external benchmark is Nuyts et al. (2024), DOI 10.1016/j.aei.2024.102443, public source https://osf.io/j7raf/; its original IFC/RDF files are not redistributed here. Reproducing the full upstream studies requires the source packages and environments described in the manuscript supplement, not just this reduced release-calculation repository.

The primary workload figure is included as editable SVG and vector PDF. The derived CSV files retain numerical precision; paper percentages are rounded to one decimal place. Uniform error placement is a counterfactual with fixed error burden, not new observed building data.

## External benchmark (separate dependencies and download)

```
python -m pip install -r requirements-external.txt
python src/fetch_external_sources.py
python src/reproduce_nuyts_and_gate.py
```

The downloader retrieves the original public OSF RDF/IFC files and checks the recorded SHA-256 values before writing them. They remain third-party data; do not add the downloaded raw files to a public Git repository. The external test writes its results under validation/results/. The 709-verdict population/reference files already included there are derived records used for the CLI check. No paid API or external-model service is used.

Network availability can change. A failure to retrieve a source is not a successful reproduction. Previously downloaded originals may be placed at the manifest paths only if their exact SHA-256 hashes match. The current local scientific re-execution used these hash-verified archived originals; fresh remote availability is recorded separately in the author QA.
