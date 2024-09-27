#!/bin/bash

## path to longcallR-dp, e.g. "/path/to/longcallR-dp"
cmd=""

## path to bam file, e.g. "path/to/aln.bam"
bam=""

## path to reference genome, e.g. "/path/to/GRCh38.fa"
ref=""

## path to truth vcf, e.g. "/path/to/truth.vcf.gz"
truth=""

## path to bed file, e.g. "/path/to/bedfile.bed"
bed=""

## path to REDIportal vcf, e.g. "path/to/REDIportal.sort.vcf.gz"
edit=""

## output_path + prefix, e.g. "sample_longcallR_train/contig"
out=""

## minimum depth, default="10"
depth=""

## minimum alternative allele fraction, default="0.05"
alt_frac=""

## minimum base quality, hifi:0, ont:10, default="0"
min_bq=""

## n_jobs, default="25"
n_jobs=""

## threads, default="8"
threads=""

## contigs, default="chr1 chr2 ... chr22, chrX chrY"
CTGS=()
for i in {1..22}; do
    CTGS+=("chr${i}")
done
CTGS+=("chrX" "chrY")

## run longcallR-dp
parallel -j ${n_jobs} \
    "$cmd --mode train \
    --bam-path '${bam}' \
    --ref-path '${ref}' \
    --editing '${edit}' \
    --truth '${truth}' \
    --bed-path '${bed}' \
    --min-depth '${depth}' \
    --min-alt-freq '${alt_frac}' \
    --min-baseq '${min_bq}' \
    --threads '${threads}' \
    --contigs {1} \
    --output '${out}_{1}'" ::: "${CTGS[@]}"


