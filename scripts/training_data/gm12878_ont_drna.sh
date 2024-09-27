#!/bin/bash

## path to longcallR-dp, e.g. "/path/to/longcallR-dp"
cmd="/hlilab/neng/projs/proj-phasing/phase/Aug7/longcallR-dp/target/release/longcallR-dp"

## path to bam file, e.g. "path/to/aln.bam"
bam="/hlilab/neng/data/LR-RNA-seq/ont/GM12878_RNA004/07_27_23_RNA004_GM12878_polyA_RNA.dorado_0.5.1_m6A_DRACH.fastq.sorted.bam"

## path to reference genome, e.g. "/path/to/GRCh38.fa"
ref="/hlilab/neng/data/genomes/GRCh38.p13/all_chr/all_chr.fa"

## path to truth vcf, e.g. "/path/to/truth.vcf.gz"
truth="/hlilab/neng/projs/proj-phasing/phase/htslib_rs/anno_eval/gm12878_truth/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"

## path to bed file, e.g. "/path/to/bedfile.bed"
bed="/hlilab/neng/projs/proj-phasing/phase/htslib_rs/anno_eval/gm12878_truth/gm12878_rna004_high_quality_region_10x.bed"

## path to REDIportal vcf, e.g. "path/to/REDIportal.sort.vcf.gz"
edit="/hlilab/neng/data/A-to-I_editing/REDIportal.sort.vcf.gz"

## output_path + prefix, e.g. "sample_longcallR_train/contig"
out="gm12878_drna_train/contig"

## minimum depth, default="10"
depth="6"

## minimum alternative allele fraction, default="0.05"
alt_frac="0.1"

## minimum baseq, hifi:0, ont:10
baseq="10"

## n_jobs, default="25"
n_jobs="5"

## threads, default="8"
threads="8"

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
    --min-baseq '${baseq}' \
    --threads '${threads}' \
    --contigs {1} \
    --output '${out}_{1}'" ::: "${CTGS[@]}"