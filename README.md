# Monolingual or multilingual instruction tuning: Which makes a better Alpaca

## Overview
This repository contains the code and data used in our work [Monolingual or multilingual instruction tuning: Which makes a better Alpaca](https://arxiv.org/abs/2309.08958) accepted to Findings of ACL: EACL 2024.

## Training
`\loraft` contains the code for low-rank adaptation (LoRA) fine-tuning. The code is derived from [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora). `\fpft` contains the code for full-parameter fine-tuning. It is based on [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Training Data
`training-data/` contains the data we used for monolingual and multilingual instruction tuning. We used the [cleaned version](https://github.com/gururise/AlpacaDataCleaned) of the original [Alpaca data](https://github.com/tatsu-lab/stanford_alpaca) in English (en) as a seed, and we used [open-source models](https://github.com/browsermt) to *machine-translate* it into 8 languages: Bulgarian (bg), Czech (cs), German (de), Spanish (es), Finnish (fi), French (fr), Russian (ru), and Chinese (zh). Note that we also make public a Portuguese (pt) version, but it was not used in our paper.

## Test Data
`test-data/` contains the test data we used to evaluate the performance of our models. We sampled English test data from [Open-Assistant](https://github.com/LAION-AI/Open-Assistant) and human-translated it into 10 languages: Bulgarian (bg), Bengali (bn), Czech (cs), Spanish (es), Finnish (fi), French (fr), Hindi (hi), Norwegian (no), Russian (ru), and Chinese (zh). Two languages, Bengali (bn) and Czech (cs), were not used in our paper.

## Trained Models/Modules
If you are interested in obtaining these, please contact us via the email provided in our paper. We are currently assessing compatibility with base models' licenses and terms and conditions before releasing trained weights.

## Citation
Please consider citing us if you use our materials.
```
@inproceedings{chen-etal-2024-monolingual,
  title={Monolingual or multilingual instruction tuning: Which makes a better {Alpaca}},
  author={Pinzhen Chen and Shaoxiong Ji and Nikolay Bogoychev and Andrey Kutuzov and Barry Haddow and Kenneth Heafield},
  year={2024},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2024},
}
```

## License
All materials created by us in this repository are licensed under [CC BY NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en).

The cleaned English data under is licensed under CC BY NC 4.0 and the original fine-tuning code is licensed under Apache 2.0 by their respective authors.

## Acknowledgments
This work stemmed from a hackathon project organized by the [High Performance Language Technologies (HPLT)](https://hplt-project.org) consortium. We are grateful to Alicia Núñez Alcover, David Samuel, Joona Kytöniemi, Jörg Tiedemann, Lucas Charpentier, Petter Mæhlum, Sampo Pyysalo, Sunit Bhattacharya, and Zhicheng Guo for project discussions, test data translation, and evaluation setup.

The work has received funding from the European Union's Horizon Europe research and innovation programme under grant agreement No 101070350, from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546], as well as from the European Research Council (ERC) under the EU's Horizon 2020 research and innovation program (agreement No 771113).

Computation in this work was performed on LUMI, Karolina, and Baskerville. We acknowledge CSC-IT Center for Science, Finland for awarding this project access to the LUMI supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CSC (Finland) and the LUMI consortium through Finnish extreme scale call (project LumiNMT). Karolina was supported by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254). The Baskerville Tier 2 HPC was funded by the EPSRC and UKRI through the World Class Labs scheme (EP/T022221/1) and the Digital Research Infrastructure programme (EP/W032244/1) and is operated by Advanced Research Computing at the University of Birmingham.
