# Subsystem verification for automated BIM compliance checking

Code release for the accompanying article, version 1.0.0.

This repository reproduces the fixed-error placement comparison on 1,350 simulated reviews (including a 650-review real-project parameter hold-out) and six authored-property batches. It includes exact-risk calculations and a counterexample showing why reusing fallible reference answers can violate a two-stage bound. The term `minimax_stratified` identifies a worst-case-risk-certified plan; coordinate descent is not a proof of global sample-size optimality.

Only code, derived non-sensitive counts, generated figures and source links are included. No active manuscript, correspondence, private author files or raw third-party IFC/RDF is included. The model provenance registry is a source-location aid, not permission to redistribute its underlying files.

The software uses fixed correct reference labels. The fallible-verifier example is a mathematical sensitivity check; no human sensitivity has been measured. The code does not certify physical or statutory compliance.

Run instructions and evidence boundaries are in REPRODUCIBLE_RUNBOOK.md. The release is available at https://github.com/Johnsonlijian/bim-subsystem-verification.
