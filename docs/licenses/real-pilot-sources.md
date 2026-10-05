# Real-data pilot sources and notices

This file records the source identities, attribution, and use limits for the private
real-data LoRA pilot. It contains no corpus examples. Keep the raw archives and records in
ignored local storage; this notice and the prepared recipe are the tracked provenance.

## DBpedia-14

- Source: [pinned `fancyzhx/dbpedia_14` repository](https://huggingface.co/datasets/fancyzhx/dbpedia_14/tree/9abd46cf7fc8b4c64290f26993c540b92aa145ac), revision `9abd46cf7fc8b4c64290f26993c540b92aa145ac`.
- Input: official `train-00000-of-00001.parquet`; SHA-256 `0640e4664a99cc94c47db1d7b2e01c14455d5bbecb8183ad1f93bde59f3f28ee`. The official test Parquet was not acquired.
- Pinned card content SHA-256: `c700cd45f4f34bd66dea0a5e71264987f5e2ed7544579765c4e59ec96da72fbd`.
- Attribution: Xiang Zhang, Junbo Zhao, and Yann LeCun, “Character-level Convolutional Networks for Text Classification,” NeurIPS 2015. The card identifies Xiang Zhang as dataset creator and points to the DBpedia 2014 source.
- Rights notice: repository metadata labels the data CC BY-SA 3.0. The pinned card says the dataset is licensed under the Creative Commons Attribution-ShareAlike License and the GNU Free Documentation License. Preserve both notices; this pilot does not resolve their relationship. Do not publish records, adapters, or weights until the ShareAlike and GFDL obligations are reviewed.

## UCI SMS Spam Collection

- Source: [UCI dataset 228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
- Input archive SHA-256: `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3`. The archive contains `SMSSpamCollection` and `readme`; only the message member was extracted.
- Extracted `SMSSpamCollection` SHA-256: `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`.
- Bundled `readme` SHA-256: `8753cd2d3cab68f80c8257851b8c2037778c267f55accb6fd1b5a2e32d36a84e`.
- Official UCI page snapshot dated 2026-10-04 SHA-256: `3bc6dae5ff7161ae2d65bcd89f353d39c16a265c5957e67b0cfffcc76746cd6f`; API metadata snapshot SHA-256: `601c4624ffae3c621c193ddcf32caef865d3c453304482a9d97f8620359ef8c8`.
- Attribution: Tiago A. de Almeida, José María Gómez Hidalgo, and Akebo Yamakami, “Contributions to the Study of SMS Spam Filtering: New Collection and Results,” ACM DOCENG 2011. The README asks researchers to cite the paper and dataset page and says contacting the authors about use is appreciated.
- Current grant: the official UCI page states CC BY 4.0. This private pilot proceeds under the current UCI corpus grant; broader release terms remain unresolved.
- Legacy notice: the archived README identifies the authors as copyright holders; provides the data “AS IS” with no warranty; assigns responsibility for use, distribution, modification, reproduction, publication, and derivative works to the user; requires indemnification of the copyright holders and affiliates; and limits their liability. Preserve the complete archive and its README. Do not treat the current UCI page as erasing these legacy notices. The source corpus combines upstream datasets whose terms were not separately audited, and the messages have not had a privacy review. Keep records and derivatives private pending separate rights and privacy review.

## Stanford Natural Language Inference Corpus

- Source: [SNLI project page](https://nlp.stanford.edu/projects/snli/) and [SNLI 1.0 archive](https://nlp.stanford.edu/projects/snli/snli_1.0.zip). Use only `snli_1.0/snli_1.0_dev.jsonl`; do not fetch or open official test data.
- Archive SHA-256: `afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e`. Development member SHA-256: `9c03faff70182ef086ebfeed2cffbabb5fcc6a84a8b3314decbbb5b01f07f4bf`. Bundled `README.txt` SHA-256: `a27e78bf4ba7bc26033fac15a6596dd41c7ff47b4909e739199c59f91cf1bd4e`. Project-page license snapshot SHA-256: `9b19a0ac70d0362d90cf59742d803cb93f3c8149d480540e4d211c91b8fdea58`.
- Attribution: Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning, “A Large Annotated Corpus for Learning Natural Language Inference,” EMNLP 2015. The README describes premise captions drawn mostly from Flickr30k and a smaller VisualGenome pilot source. It identifies `captionID` and `pairID` as provenance fields that must not be included in model input.
- Rights notice: the pinned source manifest records CC BY-SA 4.0 from the SNLI project page. The bundled README separately identifies the Flickr30k caption source as Creative Commons Attribution-ShareAlike. Preserve the source attribution and both notices. SNLI is development-only here; it is not a sealed or unseen test sample.

## Pilot-wide limits

This run is private research. Preserve these source notices with the recipe and run receipt.
Do not redistribute source records, source-derived artifacts, adapters, or weights until
the applicable terms are reviewed for the intended release. FinancialPhraseBank was
rejected and must not be used. DBpedia source-level exposure in Kev is disclosed, but row
overlap is unknown. The selected SNLI groups are new only relative to the earlier 32-group
development sample; Qwen pretraining and other model exposure are unknown. Do not claim the
pilot is contamination-free or that its examples are unseen by the models.
