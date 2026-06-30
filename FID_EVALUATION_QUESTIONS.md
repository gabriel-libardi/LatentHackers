# Questions about the FID 57.80 result

I traced through the eval pipeline behind `metrics_results` and ended up confused rather than
convinced, so flagging it here instead of assuming the worst — I may be missing context (a later
commit, a flag I didn't trace correctly).

**The "fake" images look like real retrieved plans, not generated ones.**
`generate_layout_with_stored_model` → `predict_rooms_program_aware_set` → `fit_rooms_to_outline`
(`main.py:1116-1130`, `:972-997`) fetches a *real* plan's rooms (`source_plan_id`) and warps them
onto the query outline. `refine_scaffold_with_denoiser` (`main.py:1629`) then either blends in a
25%-weighted residual or, per `METHOD.md`'s own "Why Refined Looked Identical" note, falls back to
the unmodified retrieved scaffold when its guard rejects the (undertrained) denoiser's output.

**The retrieval pool doesn't look filtered to exclude the official test set.**
`EVALUATION_PLAN_IDS` in `generazione.ipynb` comes from the official `test_indices.txt` split, but
the retrieval pool (`train_plan_ids`, `main.py:1116`) is built at `main.py:355-385` from a separate
internal site-level split over the *entire* raw dataset — `METHOD.md` itself says the official
split "is not directly mapped back to the CSV geometry workflow." The only exclusion is
`same_site_ok=False`/`same_building_ok=False` (`:935-937`), which only blocks the query's own
building, not other official test plans.

**If both hold, FID is comparing real plans to real plans.** Both REAL_DIR and FAKE_DIR are
rendered the same way, so if FAKE_DIR is mostly retrieved real geometry, FID/Density/Coverage are
measuring two real-plan distributions — which FID can't distinguish from genuine generation. That
would also explain a non-zero-but-mediocre 57.80 (different real plan retrieved each time) rather
than confirming real generative quality.

(`dataExploration/data_exploration.py:208-210` does an even more direct version of this —
`df_generated = df_real.copy()` — but its print format doesn't match `metrics_results`, so it
doesn't look like that's the source of 57.80. Flagging it anyway since it's misleading if reused.)

## Asking, not concluding

`METHOD.md` reads like an honest WIP note (diffusion model openly described as undertrained and
gated behind `neural_refinement_enabled`), which is why I don't think this is deliberate. But the
README claims this evaluates "generated layouts on the official held-out test split," and
`metrics_results` reports 57.80 with no caveat — that gap is what prompted this issue.

1. Is `FAKE_DIR` meant to hold diffusion output for this run, or is retrieval+repair the intended
   stand-in for now?
2. Should the retrieval pool be filtered to the official train split? Where would that happen?
3. Should `metrics_results`/README be caveated until generation isn't retrieval over a pool that
   includes test plans?

Happy to help fix the eval script if useful.
