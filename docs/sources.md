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
