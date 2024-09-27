use clap::{Parser, ValueEnum};
use rust_htslib;

use util::{multithread_produce3, parse_bed, Region};

use crate::util::intersect_interested_regions;

mod generate_train;
mod generate_predict;
mod util;
mod vcfs;
mod feature;

#[derive(clap::ValueEnum, Debug, Clone)]
pub enum Mode {
    Train,
    Predict,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input bam file
    #[arg(short = 'b', long)]
    bam_path: String,

    /// Path to reference file
    #[arg(short = 'f', long)]
    ref_path: String,

    /// Mode: train or predict
    #[arg(short = 'm', long)]
    mode: Mode,

    /// Truth vcf file path (Optional)
    #[arg(long)]
    truth: Option<String>,

    /// RNA editing dataset (Optional)
    #[arg(long)]
    editing: Option<String>,

    /// Passed RNA editing dataset by comparing with alignment (Optional)
    #[arg(long)]
    passed_editing: Option<String>,

    /// Interested region
    #[arg(long)]
    bed_path: Option<String>,

    /// Output bam file path
    #[arg(short = 'o', long)]
    output: String,

    /// Region to realign (Optional). Format: chr:start-end, left-closed, right-open.
    #[arg(short = 'r', long)]
    region: Option<String>,

    /// Contigs to be processed. Example: -x chr1 chr2 chr3
    #[arg(short = 'x', long, num_args(0..))]
    contigs: Option<Vec<String>>,

    /// Number of threads, default 1
    #[arg(short = 't', long, default_value_t = 1)]
    threads: usize,

    /// Minimum mapping quality for reads
    #[arg(long, default_value_t = 10)]
    min_mapq: u8,

    /// Minimum base quality for allele
    #[arg(long, default_value_t = 0)]
    min_baseq: u8,

    /// Minimum allele frequency for candidate SNPs
    #[arg(long, default_value_t = 0.1)]
    min_alt_freq: f32,

    /// Minimum alternate allele depth for candidate SNPs
    #[arg(long, default_value_t = 2)]
    min_alt_depth: u32,

    /// Minimum depth for candidate SNPs
    #[arg(long, default_value_t = 6)]
    min_depth: u32,

    /// Maximum depth for candidate SNPs
    #[arg(long, default_value_t = 20000)]
    max_depth: u32,

    /// Minimum read length to filter reads
    #[arg(long, default_value_t = 500)]
    min_read_length: usize,

    /// Flanking size
    #[arg(long, default_value_t = 20)]
    flanking_size: u32,

    /// Chunk size
    #[arg(long, default_value_t = 5000)]
    chunk_size: u32,

}

fn main() {
    let arg = Args::parse();
    let bam_path = arg.bam_path;
    let ref_path = arg.ref_path;
    let mode = arg.mode;
    let truth_path = arg.truth;
    let editing_path = arg.editing;
    let passed_editing_path: Option<String> = arg.passed_editing;
    let bed_path = arg.bed_path;
    let output_dir = arg.output;
    let input_region = arg.region;
    let input_contigs = arg.contigs;
    let threads = arg.threads;
    let min_mapq = arg.min_mapq;
    let min_baseq = arg.min_baseq;
    let min_alt_freq = arg.min_alt_freq;
    let min_alt_depth = arg.min_alt_depth;
    let min_depth = arg.min_depth;
    let max_depth = arg.max_depth;
    let min_read_length = arg.min_read_length;
    let flanking_size = arg.flanking_size;
    let chunk_size = arg.chunk_size;

    let mut regions = Vec::new();
    if input_region.is_some() {
        let region = Region::new(input_region.unwrap());
        regions = vec![region];
    } else {
        regions = multithread_produce3(
            bam_path.to_string().clone(),
            ref_path.to_string().clone(),
            threads,
            input_contigs,
            min_mapq,
            min_read_length,
        );
    }

    if bed_path.is_some() {
        let interested_regions = parse_bed(bed_path.unwrap());
        regions = intersect_interested_regions(&regions, &interested_regions, threads);
    }

    // whether output directory exists
    if !std::path::Path::new(&output_dir).exists() {
        std::fs::create_dir_all(&output_dir).unwrap();
    }

    match mode {
        Mode::Train => {
            generate_train::generate(
                bam_path,
                ref_path,
                truth_path.unwrap(),
                editing_path,
                passed_editing_path,
                output_dir,
                regions,
                min_mapq,
                min_baseq,
                min_read_length,
                min_alt_freq,
                min_alt_depth,
                min_depth,
                max_depth,
                flanking_size,
                chunk_size,
                threads);
        }
        Mode::Predict => {
            generate_predict::generate(
                bam_path,
                ref_path,
                output_dir,
                regions,
                min_mapq,
                min_baseq,
                min_read_length,
                min_alt_freq,
                min_alt_depth,
                min_depth,
                max_depth,
                flanking_size,
                chunk_size,
                threads,
            );
        }
    }
}
