# Route Choice minimal revert

`run_route_choice_lab.bat` uses `route_choice_lab_runner.py` again.

This restores the original Route Choice candidate selection and decoder while retaining:

- exact structural duplicate suppression
- route fingerprints
- Combined / Core-only / Feedback-only observability
- Core change auditing
- persistent education progress and next-cycle controls

The experimental guarded decoder remains in the repository for reference but is not used by the standard launcher.
