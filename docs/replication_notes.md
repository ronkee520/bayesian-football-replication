# Replication notes

## Paper-to-code mapping

The project follows Baio and Blangiardo (2010), but keeps two goals separate:

- reproduce the published model assumptions;
- build a stable contemporary implementation for new analysis.

`paper_replication` is therefore not silently treated as the recommended model.
`modernized` changes priors and parameterisation while preserving the score
likelihood and substantive interpretation.

## Source discrepancies recorded explicitly

The paper body and BUGS appendix do not use identical hyperprior values. The body
describes Gamma(0.1, 0.1) precisions for the basic model, while the appendix and
mixture code use Gamma(0.01, 0.01) in places. This repository uses Gamma(0.1, 0.1)
for the basic paper profile and Gamma(0.01, 0.01) for the mixture paper profile.

For the mixture model, the middle component is fixed at zero, following the BUGS
appendix. The prose in the paper can instead be read as assigning it another
normal level. Fixing the middle component gives a clear weak/middle/strong anchor.

The paper's Section 2 illustration uses Serie A 1991/92. Section 4 re-fits both
the basic and mixture models to Serie A 2007/08 for the comparison in Table 3.
The two models can and should be compared on the same EPL data in this project.

## Deliberate modernizations

- NumPyro NUTS replaces hand-written random-walk Metropolis algorithms.
- Discrete mixture allocations are marginalized.
- The modernized basic model uses symmetric non-centred team effects.
- Regularising Half-Normal scale priors replace diffuse Gamma precision priors.
- Diagnostics come from ArviZ and include rank-normalized R-hat and bulk/tail ESS.
- Forecast claims require a time-ordered holdout, not full-season re-prediction.

## Interpretation cautions

The Poisson likelihood treats home and away goals as conditionally independent.
It can miss residual dependence and low-score draw inflation. Administrative
point deductions are outside the match-score data-generating process. All league
tables in the code are therefore labelled as on-field points.
