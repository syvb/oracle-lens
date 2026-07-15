# Vendored third-party code

Both projects are frozen, unmaintained reference implementations with no PyPI
releases, so they are vendored here verbatim (Apache-2.0, licenses included in
each tree) rather than pinned as git dependencies. **Do not modify vendored
code** — all adaptations live in `oracle_lens` via subclassing/wrapping, so a
`diff` against upstream stays empty.

| Directory | Upstream | Pinned commit |
|---|---|---|
| `natural_language_autoencoders/` | https://github.com/kitft/natural_language_autoencoders | `1b7f13d9d8a37075cd2e5d1604eca57820216ed5` |
| `jacobian-lens/` | https://github.com/anthropics/jacobian-lens | `581d398613e5602a5af361e1c34d3a92ea82ba8e` |

Notes:
- https://github.com/kitft/nla-inference is NOT vendored: it is a byte-identical
  subset of `natural_language_autoencoders` (its `nla_inference.py` and examples
  ship at that repo's root).
- What we import from `nla` (the pure, Miles-independent modules): `injection`,
  `schema`, `config`, `models`, `arch_adapters`, `datagen.extractors`, plus the
  root-level `nla_inference.py` client. The Miles/SGLang-bound training stack
  (`train_actor`, `rollout/`, `reward`, the `loss` driver) is reference-only
  until/unless M5 (GRPO) is green-lit, at which point Miles + SGLang are
  installed per `natural_language_autoencoders/docs/setup.md` with the patches
  shipped in that tree.
- From `jlens` we use `fit`/`apply`/`transport`/`unembed` (layer check + eval
  baseline) and the slice visualizer.
