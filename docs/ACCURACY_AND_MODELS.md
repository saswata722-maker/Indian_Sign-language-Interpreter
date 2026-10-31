# Accuracy fixes and pretrained-model research

Reviewed 2026-10-31. These are local source changes; no new production weights
have been trained and no pretrained replacement has been deployed.

## Verification performed

- 15 regression/integration tests passed, including a synthetic one-epoch
  training run with checkpoint and split-manifest round trip.
- The shipped checkpoint still scores 756/852 (88.73%) on reconstructed legacy
  validation after the refactor. No legacy feature-format regression was observed.
- Two held-out source videos were replayed through fresh MediaPipe instances at
  a 10-FPS sampling cadence: both full clips classified correctly; one final
  streaming window classified correctly and the shorter clip was still buffering.
  This is a smoke check only, not a meaningful new accuracy estimate.
- Python compilation, browser JavaScript syntax, and Git whitespace checks passed.
- No physical webcam session or cloud deployment was exercised. New signer
  accuracy, automatic segmentation quality, and production throughput remain
  unmeasured. The V2 model needs a full training run.

## What was fixed

- Desktop, web, and video replay use `inference/runtime.py` and versioned
  `preprocessing/features.py` instead of separate inference implementations.
- Buffers are bounded by both timestamps and a frame-count ceiling, even on
  gate rejection. Irregular capture timestamps are used for temporal resampling.
- The browser permits only one in-flight request. Requests carry sequence IDs
  and capture timestamps; the API rejects out-of-order frames. Every client has
  its own MediaPipe tracker and reset state. Inactive clients are cleaned up.
- Default interactive mode records one complete sign before classification.
  Continuous preview is still available, but it is not a trained continuous
  sign recognizer. Desktop SPACE records/finishes; S changes mode.
- Low confidence clears the displayed prediction. Probability averaging reports
  confidence for the actual displayed class; stale averaging history expires.
- Motion gating considers hand XY coordinates with valid consecutive tracking,
  excluding detection flags and missing/detected transitions. Motion is normalized
  to a 25-FPS reference step. Manually delimited signs are not motion-rejected.
  The new motion statistic needs validation on webcam data; the old numeric gate
  threshold was not calibrated for this statistic.
- Spatial augmentation respects XYZ triples, detection flags, and absent body
  parts. Time warps are monotonic. V2 resampling preserves binary masks, missing
  coordinates stay zero, and velocity does not bridge tracking dropouts.
- The shipped checkpoint has no feature-version field, so inference uses V1.
  New training uses V2 and writes `models_v2/`; the original checkpoint is retained.
- Training saves train/validation/test manifests inside checkpoints. Evaluation
  defaults to validation; independent test evaluation requires a saved manifest.
  Old checkpoints reconstruct validation with an explicit dataset-order warning.
- Pipeline replay samples held-out clips across the dataset, resets tracking
  between clips, and reports full-clip and final-window results separately. It
  does not select the best prediction retrospectively from a stream.

## Migration and honest evaluation

The old model's saved validation accuracy is 88.73%. The 97.51% report included
training examples. Neither measures new-signer webcam accuracy.

Existing legacy `.npy` files can be used for V2 training: missing coordinates
are zeroed using their flags before resampling. However, old extraction reused
tracking across unrelated videos. For the cleanest retraining run, extract into
a new directory (the extractor now resets at each video), set `landmarks_dir`
in `config.yaml` to that directory, and train with a fresh saved split manifest.
Do not mix old and new cache files. Changing normalization beyond the versioned
masking implemented here requires another feature version and new training.

For signer-independent evaluation, provide a JSON mapping from relative landmark
paths to signer IDs (or recording-session IDs if evaluating session transfer):

```json
{"Greetings/48. Hello/clip001.npy": "signer_01"}
```

Supply all file paths, then run `python -m utils.make_splits --groups groups.json`.
Without real metadata, the tool deliberately calls the split `clip-random`.
It will not invent signer IDs from file numbering. Keep all crops/augmentations
from a recording in the same group. Existing `splits.json` is never overwritten.
Keep the final test partition fixed; choose thresholds using validation only.

An independent webcam test should include complete signs, idle periods, other
hand movements, and transitions, with labels and start/end timestamps. Report
sign-level accuracy, rejection, false recognitions per minute, and latency.
The software fixes do not establish a new live accuracy percentage. Automatic
sign-boundary detection and learned background rejection remain data-dependent
follow-up work; the manual segment mode provides a useful baseline now.

## Pretrained models worth benchmarking

### 1. OpenHands INCLUDE DPC / SL-GCN: closest practical match

The authors publish INCLUDE-specific `include_dpc.zip`, `include_slgcn.zip`,
`include_stgcn.zip`, `include_bert.zip`, and `include_lstm.zip` release assets.
The GitHub release API confirmed the DPC archive is 35,639,064 bytes and the
SL-GCN archive is 40,608,718 bytes. These are existing release assets, not models
I trained or evaluated here.

- [Official checkpoint release](https://github.com/AI4Bharat/OpenHands/releases/tag/checkpoints_v1)
- [INCLUDE DPC checkpoint](https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_dpc.zip)
- [INCLUDE SL-GCN checkpoint](https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/include_slgcn.zip)
- [ACL 2022 paper](https://aclanthology.org/2022.acl-long.150/)
- [Author fine-tuning config](https://github.com/AI4Bharat/OpenHands/blob/main/examples/configs/include/pose_finetune_dpc.yaml)

The DPC recipe selects `mediapipe_holistic_minimal_27`, applies its own shoulder
normalization, and uses sequences arranged as six groups of ten frames. Our
75-joint, XYZ-plus-flags-plus-velocity tensor cannot be passed directly to it.
Reproduce the author's joint selection, ordering, normalization, sampling, and
class metadata, then evaluate with the same webcam protocol. For transfer,
replace/remap the classifier and fine-tune on ISL webcam recordings. The project
is Apache-2.0 but its repository states it is no longer actively maintained;
use a separate environment and check checkpoint/data terms before redistribution.
An INCLUDE-trained public checkpoint may have seen clips in a locally created
random test split. Use its official split or genuinely new webcam data.

### 2. SignCLIP: multilingual pose representation and few-shot adaptation

- [EMNLP 2024 paper](https://aclanthology.org/2024.emnlp-main.518/)
- [Author code and model-weight links](https://github.com/J22Melody/fairseq/tree/main/examples/MMPT#demo-and-model-weights)

The authors release multilingual and language-adapted models and provide a
minimal Python 3.12 inference environment. The paper studies multilingual
pose/text pretraining and downstream adaptation. A useful experiment is frozen
pose embeddings plus an ISL linear classifier, followed by fine-tuning. Follow
its pose-format preprocessing and language conventions; do not assume English
word prompts make ASL signs equivalent to Indian signs. Research/development
dependencies are older and should stay isolated from this application's runtime.

### 3. Uni-Sign: larger pose/RGB transfer experiment

- [ICLR 2025 paper](https://arxiv.org/abs/2501.15187)
- [Official repository](https://github.com/ZechengLi19/Uni-Sign)
- [Author checkpoint repository](https://huggingface.co/ZechengLi19/Uni-Sign)

The authors provide pose-only and RGB/pose training stages, downstream fine-tuning,
checkpoints, and an online demo. This is a candidate for exploiting image detail
when landmarks are unreliable. It is a larger architectural migration than
OpenHands and is not an ISL-ready 262-class checkpoint. Measure memory and latency
on the intended hardware, adapt the output task, and fine-tune on ISL. The author
checkpoint link was verified in the README; binary weights were not downloaded.

### 4. NLA-SLR: visually confusable signs and RGB/pose fusion

- [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zuo_Natural_Language-Assisted_Sign_Language_Recognition_CVPR_2023_paper.html)
- [Official code and checkpoint table](https://github.com/FangyunWei/SLRT/tree/main/NLA-SLR)

The paper combines video/keypoint information and language-assisted training to
address visually similar signs. The repository publishes WLASL/MSASL checkpoints.
This is relevant to confusion pairs but requires a different input pipeline and
ISL adaptation. Its reported dataset scores are not comparable to our live score.

### Additional ISL data research

[CISLR (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.707/) introduces
approximately 4,700 word classes and studies transfer from resource-rich ASL
for one-shot ISL recognition. It supports investigating cross-language features,
but combining datasets requires checking gloss equivalence, sign variants,
permissions, and overlap. More classes alone do not resolve webcam domain shift.

## Recommended experiment order

1. Measure the shipped model with the corrected complete-sign workflow on new
   labeled webcam clips; retain recordings and capture timestamps.
2. Retrain the corrected V2 baseline with fixed partitions and webcam examples.
3. Reproduce OpenHands INCLUDE DPC/SL-GCN with their exact preprocessing. Compare
   on the same external webcam set, not potentially overlapping INCLUDE clips.
4. Benchmark SignCLIP embeddings; explore Uni-Sign or NLA-SLR only if the measured
   accuracy/latency tradeoff justifies the extra deployment complexity.

No candidate is claimed to be better on this project's live camera without that
comparison. No external model has been substituted silently.
