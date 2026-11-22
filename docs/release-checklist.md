# Release review

- Choose the license for the project-owned source and wrapper. Add the final paper citation.
- Obtain redistribution clearance for author-received TmF and DER, N20 PrivGraph and the pinned DPG author modules if these files are to accompany the repository. The current Git candidate uses externally bound sources and verifies their hashes. Retain the GPL and author notices with any distributed PrivHRG adaptation.
- Confirm the original execution pin for the separately hashed PrivHRG Python source, which preserves the calibrated finite-chain variant.
- Confirm the arithmetic-compatibility branches used in the historical DER runs. The supplied route invokes the frozen author_uint64 branch. Alternate numeric paths are recorded in the harness. Their exact source and run binding need separate verification.
- The CPU wrappers for the two DPG models differ from original CPU/CUDA staging and acceleration. Confirm the original device, feature-extraction and batching paths if exact process equivalence is required. The local PrivCGS C++ build consumes the pinned source. Confirm binary or numerical equivalence if claiming identical saved graphs across platforms.
- The inherited PrivCGS bundle hashes differ from the deployed record for `standalone/lib/route_s_clean/_native_src/conditional_transport/packed_transport.cpp` and `standalone/lib/route_s_clean/_native_build.py`. The deployment freeze pins both files. The native manifest and approved failure-only continuation record independently identify the transport source. Both versions are recorded in [source evidence](../configs/source-evidence.json).
- Check exact original node ordering against the frozen input digest for each dataset. The launcher distinguishes an ordinary preparation from a verified frozen input. Confirm download terms for the specific Flickr-G and other data releases.

Static parsing, source hashes, launch plans and representative build or import checks are appropriate for reviewing this package. The release contains no datasets, fitted models, benchmark outputs or experiment logs.
