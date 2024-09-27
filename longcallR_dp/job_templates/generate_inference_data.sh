#!/bin/bash

## path to longcallR-dp, e.g. "/path/to/longcallR-dp"
cmd=""

## path to bam file, e.g. "path/to/aln.bam"
bam=""

## path to reference genome, e.g. "/path/to/GRCh38.fa"
ref=""

## output_path + prefix, e.g. "sample_longcallR_inference/contig"
out=""

## minimum depth, default="6"
depth=""

## minimum alternative allele fraction, default="0.1"
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
    "$cmd --mode predict \
    --bam-path '${bam}' \
    --ref-path '${ref}' \
    --min-depth '${depth}' \
    --min-alt-freq '${alt_frac}' \
    --min-baseq '${min_bq}' \
    --threads '${threads}' \
    --contigs {1} \
    --output '${out}_{1}'" ::: "${CTGS[@]}"


