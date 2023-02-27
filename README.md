## SLTUNET: A Simple Unified Model for Sign Language Translation (ICLR 2023)

Note we will release our source code as well as trained models to facilitate follow-up studies! The current version is rough, and we will further polish it before our release.

The modeling part is mainly in `models/transformer.py`, including shared/modality-specific encoder, shared decoder and the joint training objectives.

- Regarding how we train different tasks, please see `train_fn` in `transformer.py`.


