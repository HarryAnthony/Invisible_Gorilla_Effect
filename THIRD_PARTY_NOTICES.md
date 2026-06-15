# Third-Party Notices

This repository includes vendored or adapted code from third-party projects. The
original project license applies to each component listed below. The MIT license
in the root [`LICENSE`](LICENSE) file applies only to original work in this
repository by Harry Anthony, not to these bundled components.

| Component | Location | Upstream | License |
|-----------|----------|----------|---------|
| zennit-crp (trimmed) | `source/post_hoc_methods/post_hoc_utils/crp/` | [rachtibat/zennit-crp](https://github.com/rachtibat/zennit-crp) | Clear BSD License — see [`crp/LICENSE`](source/post_hoc_methods/post_hoc_utils/crp/LICENSE) |
| zennit | `source/post_hoc_methods/post_hoc_utils/zennit/` | [chr5tphr/zennit](https://github.com/chr5tphr/zennit) | GNU LGPL v3 or later — see [`zennit/COPYING.LESSER`](source/post_hoc_methods/post_hoc_utils/zennit/COPYING.LESSER) and [`zennit/COPYING`](source/post_hoc_methods/post_hoc_utils/zennit/COPYING) |
| bayesian-torch (subset) | `source/ad_hoc_methods/ad_hoc_utils/bayesian_torch/` | [IntelLabs/bayesian-torch](https://github.com/IntelLabs/bayesian-torch) | BSD-3-Clause — see file headers and [`bayesian_torch/ReadMe`](source/ad_hoc_methods/ad_hoc_utils/bayesian_torch/ReadMe) |

## Notes

- **PCX** (`source/post_hoc_methods/PCX.py`) uses the trimmed CRP and zennit copies above.
- **BNN** ad-hoc evaluation uses the bundled bayesian-torch subset.
- Other dependencies (PyTorch, scikit-learn, etc.) are installed via pip and are
  not redistributed in this repository; refer to each package's own license.
