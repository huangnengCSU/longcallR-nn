# <longcallR-nn>

## Table of Contents
- [Introduction](#introduction)
- [Install](#install)
- [Usage](#usage)

## Introduction

<longcallR_nn> is a variant caller specifically designed for long-read RNA-seq data, utilizing a ResNet50 model. It provides high-precision identification of RNA SNPs and A-to-I RNA editing events. Currently, longcallR_nn is compatible with PacBio MAS-Seq/Iso-Seq and Nanopore cDNA/dRNA data.

## Installation

### Prerequisites
- [Python, Rust]

### Install
1. Clone the repo:
    ```bash
    git clone https://github.com/huangnengCSU/longcallR-nn.git
    ```
2. Navigate to the project directory:
    ```bash
    cd longcallR-nn
    ```
3. Install the longcallR-nn dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Install longcallR-nn:
    ```bash
    pip install .
    ```
5. Compile longcallR-dp:
    ```bash
    cd longcallR_dp
    cargo build --release
    ```

## Usage

### longcallR-dp
Template scripts for generating training dataset or inference dataset can be found in the `longcallR_dp/job_templates` directory.
```bash
Usage: longcallR_dp [OPTIONS] --bam-path <BAM_PATH> --ref-path <REF_PATH> --mode <MODE> --output <OUTPUT>

Options:
  -b, --bam-path <BAM_PATH>
          Path to input bam file
  -f, --ref-path <REF_PATH>
          Path to reference file
  -m, --mode <MODE>
          Mode: train or predict [possible values: train, predict]
      --truth <TRUTH>
          Truth vcf file path (Optional)
      --editing <EDITING>
          RNA editing dataset (Optional)
      --passed-editing <PASSED_EDITING>
          Passed RNA editing dataset by comparing with alignment (Optional)
      --bed-path <BED_PATH>
          Interested region
  -o, --output <OUTPUT>
          Output bam file path
  -r, --region <REGION>
          Region to realign (Optional). Format: chr:start-end, left-closed, right-open
  -x, --contigs [<CONTIGS>...]
          Contigs to be processed. Example: -x chr1 chr2 chr3
  -t, --threads <THREADS>
          Number of threads, default 1 [default: 1]
      --min-mapq <MIN_MAPQ>
          Minimum mapping quality for reads [default: 10]
      --min-baseq <MIN_BASEQ>
          Minimum base quality for allele [default: 0]
      --min-alt-freq <MIN_ALT_FREQ>
          Minimum allele frequency for candidate SNPs [default: 0.1]
      --min-alt-depth <MIN_ALT_DEPTH>
          Minimum alternate allele depth for candidate SNPs [default: 2]
      --min-depth <MIN_DEPTH>
          Minimum depth for candidate SNPs [default: 6]
      --max-depth <MAX_DEPTH>
          Maximum depth for candidate SNPs [default: 20000]
      --min-read-length <MIN_READ_LENGTH>
          Minimum read length to filter reads [default: 500]
      --flanking-size <FLANKING_SIZE>
          Flanking size [default: 20]
      --chunk-size <CHUNK_SIZE>
          Chunk size [default: 5000]
  -h, --help
          Print help
  -V, --version
          Print version
```

### longcallR-nn call variants
The pretrained configuration files can be found in the `config` directory, while the model files are located in the `models` directory.
```bash
longcallR_nn call [-h] -config CONFIG -model MODEL -data DATA [-ref REF] -output OUTPUT [-max_depth MAX_DEPTH] [-batch_size BATCH_SIZE] [--no_cuda]

optional arguments:
  -h, --help                show this help message and exit
  -config CONFIG            path to config file
  -model MODEL              path to trained model
  -data DATA                directory of feature files
  -ref REF                  reference genome file
  -output OUTPUT            output vcf file
  -max_depth MAX_DEPTH      max depth threshold
  -batch_size BATCH_SIZE    batch size
  --no_cuda                 If running on cpu device, set the argument.
```

#### Table 1: config and model for pretrained model
| Platform | config | model |
|----------|----------|----------|
| PacBio Masseq  | config/hg002_na24385_masseq.yaml  | models/hg002_na24385_mix_nopass_resnet50_sgd.epoch30.chkpt  |
| PacBio Isoseq  | config/hg002_isoseq.yaml  | models/hg002_baylor_isoseq_nopass_resnet50_sgd.epoch30.chkpt  |
| ONT cDNA  | config/wtc11_cdna.yaml  | models/cdna_wtc11_nopass_resnet50_sgd.epoch30.chkpt  |
| ONT dRNA  | config/gm12878_drna.yaml  | models/drna_gm12878_nopass_resnet50_sgd.epoch30.chkpt  |


### longcallR-nn train model
The template configuration file to refer to is `config/test.yaml`.
```bash
longcallR_nn train [-h] -config CONFIG [-log LOG]

optional arguments:
  -h, --help      show this help message and exit
  -config CONFIG  path to config file
  -log LOG        name of log file
```

