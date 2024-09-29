# <longcallR-nn>

## Quick Run
```bash
## path to longcallR-dp, e.g. "/path/to/longcallR-dp"
cmd="/path/to/longcallR-dp"
## path to bam file, e.g. "path/to/aln.bam"
bam="demo/hg002_chr22.bam"
## path to reference genome, e.g. "/path/to/GRCh38.fa"
ref="demo/chr22.fa"
## output_path + prefix, e.g. "sample_longcallR_inference/contig"
out="hg002_chr22_feature/contig"
## minimum depth, default="6"
depth="6"
## minimum alternative allele fraction, default="0.1"
alt_frac="0.1"
## minimum base quality, hifi:0, ont:10, default="0"
min_bq="0"
## n_jobs, default="25"
n_jobs="1"
## threads, default="8"
threads="4"
## contigs, default="chr1 chr2 ... chr22, chrX chrY"
CTGS=("chr22")


## run longcallR-dp (Parallel)
parallel -j ${n_jobs} \
    "$cmd --mode predict \
    --bam-path '${bam}' \
    --ref-path '${ref}' \
    --min-depth '${depth}' \
    --min-alt-freq '${alt_frac}' \
    --min-baseq '${min_bq}' \
    --threads '${threads}' \
    --contigs {1} \
    --output '${out}_{1}'" ::: "${CTGS[@]}"


## run longcallR-nn (Serial)
for ctg in "${CTGS[@]}"; do
    longcallR_nn call \
    -config config/hg002_na24385_masseq.yaml \
    -model models/hg002_na24385_mix_nopass_resnet50_sgd.epoch30.chkpt \
    -data ${out}_${ctg} \
    -ref ${ref} \
    -output hg002_chr22.vcf \
    -max_depth 200 \
    -batch_size 500
done
```

## Table of Contents
- [Introduction](#introduction)
- [Install](#install)
- [Usage](#usage)
- [License](#license)

## Introduction

<longcallR_nn> is a variant caller specifically designed for long-read RNA-seq data, utilizing a ResNet50 model. It provides high-precision identification of RNA SNPs and A-to-I RNA editing events. Currently, longcallR_nn is compatible with PacBio MAS-Seq/Iso-Seq and Nanopore cDNA/dRNA data.

## Installation

### Prerequisites
- [Python, [Rust](https://www.rust-lang.org/), Parallel]

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
    # create env
    conda create -n longcallRenv python=3.9
    # activate env
    conda activate longcallRenv
    # install dependencies
    pip install -r requirements.txt
    # select one of the following commands to install PyTorch (version 1.3.0 or higher)
    conda install pytorch torchvision torchaudio torchmetrics pytorch-cuda=<CUDA_VERSION> -c pytorch -c nvidia
    conda install pytorch torchvision torchaudio torchmetrics cpuonly -c pytorch
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

### longcallR-dp extract training or inference dataset
Template scripts for generating training dataset or inference dataset can be found in the `longcallR_dp/job_templates` directory. The script for generating a [REDIportal](http://srv00.recas.ba.infn.it/atlas) VCF file for training A-to-I RNA-editing events is located at `longcallR_dp/scripts/REDIportal_to_vcf.py`.
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


## License
MIT License

Copyright (c) 2024 Dana-Farber Cancer Institute.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

