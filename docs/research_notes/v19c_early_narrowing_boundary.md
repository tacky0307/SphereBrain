# SphereBrain Research Note — v19c Early Narrowing Boundary

Goal:
Determine the minimum broad exploration period required before narrowing pathways.

## Fixed
- residual flow = 0.000

## Compare
- Boundary-1
- Boundary-2
- Boundary-3
- Boundary-4
- Boundary-5
- Boundary-6

Metrics:
- Top-1 recall
- Successor margin
- Similarity@24
- Recall stabilization step

Hypothesis:
Broad exploration is only needed for the first few propagation steps. After that, narrowing should preserve recall while preventing convergence.
