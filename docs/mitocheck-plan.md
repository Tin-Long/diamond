# Offline MitoCheck DIAMOND adaptation

Goal: compare the image-space DIAMOND world model using only the fixed paired trajectories in mitov2. Implement inline in the requested diamond directory; leave Atari defaults intact.

- Fixed membership: split_index.sqlite tracks(pair_eligible=1), 6571 train/1642 test. Resolve current images and metadata from training_ready_v2/index.sqlite because historical root image paths were removed.
- Select only eligible single cells belonging to those tracks. Split missing frames into contiguous segments. No composites, marginal-only samples or cross-track windows.
- Images: uint8 grayscale 96x96, deterministic [-1,1] scaling. Default history=1. Gene2Vec 200 dimensions and source/target absolute time (19 + frame*0.5) plus source age. Validation: deterministic gene-stratified 5% of training tracks.
- Keep EDM denoising and original sampler; optional continuous-condition projection in InnerModel; preserve discrete Atari actions. Cell loss trains one target, with clean image history.
- Add Hydra offline train/audit and held-out evaluation entries, portable shell launcher, lightweight cell-only requirements. Save memberships and data fingerprints, preprocessing/config, optimizer, RNG and resume state.
- Test synthetic split exclusion, gaps/history, normalization/conditions, loss/gradient/sampling, legacy Atari mode, checkpoint resume and image-space feedback. Audit real local data; production GPU run remains server-side.
- ChatGPT independently reviews files and released test outputs after execution.
