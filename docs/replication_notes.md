# Replication notes

## Paper-to-code mapping

The implementation follows Baio and Blangiardo (2010) while separating two specifications:

- reproduce the published model assumptions;
- evaluate a contemporary parameterisation under the same score likelihood.

`paper_replication` records the paper-oriented assumptions. `modernized` changes
the priors and parameterisation while preserving the score likelihood and the
interpretation of team effects.

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
Both models are compared on the same EPL data in this repository.

## Implementation differences

- NumPyro NUTS replaces hand-written random-walk Metropolis algorithms.
- Discrete mixture allocations are marginalized.
- The modernized basic model uses symmetric non-centred team effects.
- Regularising Half-Normal scale priors replace diffuse Gamma precision priors.
- Diagnostics come from ArviZ and include rank-normalized R-hat and bulk/tail ESS.
- Forecast performance is evaluated with a time-ordered holdout rather than
  full-season re-prediction.

## Interpretation cautions

The Poisson likelihood treats home and away goals as conditionally independent.
It can miss residual dependence and low-score draw inflation. Administrative
point deductions are outside the match-score data-generating process. All league
tables in the code are therefore labelled as on-field points.
