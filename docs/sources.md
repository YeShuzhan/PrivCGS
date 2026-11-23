# Sources and redistribution

The root README follows the installation, dataset and run-command structure of [PrivSTD](https://github.com/YeShuzhan/PrivSTD). [Implementation routes](../configs/implementations.json) list runnable file sets and SHA-256 hashes. [Source evidence](../configs/source-evidence.json) records the freeze and validation hashes used in this package.

| Method | Source and acquisition | Selected adaptation |
| --- | --- | --- |
| PrivCGS | [Project repository](https://github.com/YeShuzhan/PrivCGS), project-owned native and Python source in `methods/privcgs`. | `linux_entry.py` for the release and `full_entry.py` for Full TMMR assignment and joint-risk refinement. |
| LDPGen | Qin et al., CCS 2017. [Paper](https://doi.org/10.1145/3133956.3134086). Based on separately received author source. | Bundled signed-total centralized Gaussian adaptation. |
| TmF | Nguyen, Imine and Rusinowitch, ASONAM 2015. [Paper](https://doi.org/10.1145/2808797.2809385). Acquire the author-received `baseline_graph.zip` and the pinned adaptation from the replication authors. | Gaussian scales in an author-lineage flow. The large extension binds the verified sparse storage adapter. |
| DER | Chen, Fung, Yu and Desai, VLDB Journal 2014. [Paper](https://doi.org/10.1007/s00778-013-0344-8). Acquire the received archive and exact `fixh=6` adaptation from the replication authors. | Gaussian component calibration and author scheduling, with separately pinned arithmetic variants where retained receipts support them. |
| PrivHRG | Xiao, Chen and Tan, KDD 2014. [Paper](https://doi.org/10.1145/2623330.2623642), [author C++ repository](https://github.com/kaseyxiao/privHRG), lineage commit `74e0a7b0666a6ca7851d4eef2ef6fa4b1d3a7ccf`. We adapted the baseline implementation supplied by the PrivGraph authors. | The Python reproduction profile retains finite MCMC tree fitting and uses Gaussian region counts. Request the profile and its GPL-2.0-or-later notices from the maintainers. |
| PrivGraph | Yuan et al., USENIX Security 2023. [Paper](https://www.usenix.org/system/files/usenixsecurity23-yuan-quan.pdf), [author repository](https://github.com/Privacy-Graph/PrivGraph), upstream reference `01a9f1164300887c56eb3fd8da66d78c9e894be9`. | N20 Gaussian calibration and inherited exponential community adjustment. An upstream checkout alone does not provide the adapted snapshot. |
| Tricycle | Jorgensen, Yu and Cormode, SIGMOD 2016. [DPAGM paper](https://doi.org/10.1145/2882903.2915215). | Gaussian Ladder for the triangle target, with Chung–Lu initialization and FIFO repair. See the [implementation](../methods/tricycle-gaussian-ladder). |
| DPGVAE | Yang et al., IJCAI 2021. [Paper](https://www.ijcai.org/proceedings/2021/450), [author repository](https://github.com/haonan3/Secure-Network-Release-with-Link-Privacy), commit `cbc28d38346921bf06147ed7a63494132f3c207d`. | Author model and configs with our [CUDA trainer](../methods/dpg-cuda) and Gaussian calibration. |
| DPGGAN | [Shared author paper](https://www.ijcai.org/proceedings/2021/450) and the same pinned commit. | Adversarial model with the same CUDA training framework and Gaussian calibration. |
| PrivDPR | Zhang, Hu, Ye and Xu, KDD 2025. [Paper](https://doi.org/10.1145/3690624.3709334), [author preprint](https://arxiv.org/abs/2501.02354). | Our [PyTorch implementation](../methods/privdpr-cuda/privdpr_zcdp.py), exposed through the `privdpr-cuda` profile. |

The received archive `baseline_graph.zip` has recorded SHA-256 `a14954721e53f569604236303fdc592b0d4eb6f2fe532acea0cc85bcb4926cce`. Its unadapted TmF file has SHA-256 `8e2a11ce7440894ab80824fc700b3a6f76066dadf1d90be9b00e2ab772401ff6`. The archive itself, inputs, precompiled shared libraries and unverified third-party source are kept outside the repository. The experiment archive includes research branches and trial settings. Give precedence to exact bindings and receipts over generic filenames or trial dates.

## Bind an external source bundle

Run `python run.py sources --method METHOD` to list profiles. Follow the linked author repository and commit when available. For a local adaptation or author-received material, request the source bundle for the listed profile under its applicable terms. Place each required file at the listed `path` under `--source-root`. Run `python run.py sources --method METHOD --profile PROFILE --source-root external --verify` to compare the acquired files with the profile hashes. A matching hash establishes the contents of the selected package. Historical run correspondence also requires the command, runtime and device recorded for that run.

For DPGVAE and DPGGAN the pinned author checkout provides the externally bound modules directly.

```bash
git clone https://github.com/haonan3/Secure-Network-Release-with-Link-Privacy ../dpg-upstream
mkdir -p external/dpg-cuda/vendor/dpg_author
git -C ../dpg-upstream archive cbc28d38346921bf06147ed7a63494132f3c207d src \
  | tar -x -C external/dpg-cuda/vendor/dpg_author
python run.py sources --method DPGVAE --source-root external --verify
```

Keep the author checkout outside this repository. For N20 PrivGraph, TmF and DER, request the adapted file set shown by the same verification command. The author originals document the lineage and can have different bytes.

## PrivCGS source version and run check

The archived Full Facebook result at `epsilon=1`, `seed=42` references the original Windows experiment's `CODE_HASHES.json` (SHA-256 `96cbfbf67b5155609e2778744851a8bec9e5081847aa4e2501bcaf264e77a25a`). The public package follows the later Linux deployment, whose `FROZEN_VERSION.json` is pinned in [source evidence](../configs/source-evidence.json). The packaged bytes match that deployment for 106 of the 110 Full bindings, including both files below. The four packaging changes are `full_entry.py` (the Full launcher derived from `study_entry.py`), `generation.py`, `linux_entry.py`, and `standalone/release.py`.

| File | Original Full experiment SHA-256 | Linux deployment and public package SHA-256 |
| --- | --- | --- |
| [`packed_transport.cpp`](../methods/privcgs/standalone/lib/route_s_clean/_native_src/conditional_transport/packed_transport.cpp) | `20f993c229535651275e8f9ee3396b15dc63dd4d83cf773266f3738a1b585699` | `a3db0187da88d6540c74892c41d4db5f3a38a23db971a5e19fbd31bbee402f7f` |
| [`_native_build.py`](../methods/privcgs/standalone/lib/route_s_clean/_native_build.py) | `2ea0c3e59e1be142d395ce093f98d8ff4cbff75e6bbeb808248f40938b1ceaf2` | `4e41e4bdaca40a6cd151f506b7a5be42d6ced759aa685ebc6edd75117e4b7522` |

The transport update adds a continuation after the legacy solver fails its convergence check. The newer C++ hash is recorded in both the Linux native-library manifest and the deployment freeze. The inherited bundle still records the original transport hash. The Python build helper adds Linux compiler flags and support for loading a verified deployment library. The native-library manifest records the earlier build helper (`43fcb51fa450a1e72ad8f62bf9b79b40b7036944f4ca9507a70303c076533e77`), and the deployment freeze records the later helper. The public package compiles the native code locally. Compiler, library and threading settings can affect numerical outputs.

On Linux x86-64 with Python 3.9.13 and GCC 13.3.0, the [README Facebook commands](../README.md#3-datasets) prepared the public SNAP data and completed a fresh Full run with the default settings. The prepared input and the archived Full input contained the same 88,234 labeled undirected edges. Their file bytes have different hashes. The new run exited with code 0. Its edge list had 88,245 unique undirected edges on the 4,039-node domain, with no self-loops and an exact match to the saved target degree sequence. The output edge-list SHA-256 was `05c81bef5605d62151190b3f81320e6c648c57f7e7374556ca7fc16ae2945423`.
