# Attribution and provenance

Jean Paul Ruiz adapted the overall fine-tuning workflow from a guide credited locally to coderay2020, then wrote the revised agent runbook, five implementation scripts, and evaluation workflow. This release credits the source author without redistributing the Google document, its PDF/Markdown exports, or its illustrations. The source was shared with the maintainer; no license for the source document was established.

The runbook's “What changed from the original” section records the major technical corrections. Source and implementation were compared before release: none of the five complete scripts occurs as a full block in the source guide. The code and runbook are released under MIT for Jean Paul Ruiz's contributions; this license does not apply to the archived source materials, private examples, or model weights.

Historical CPU smoke tests used `trl-internal-testing/tiny-Qwen2ForCausalLM-2.5` and very small evaluation samples. No task-quality improvement has been demonstrated. Private datasets and adapter weights are excluded. One recovered historical `ft-test` record did not match the later local training/validation files; neither that record nor those files are included here. The better-supported `from-runbook` snapshot supplied the scripts in this repository. The published scripts match that snapshot byte for byte.
