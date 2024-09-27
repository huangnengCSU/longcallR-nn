use std::{fs, fs::File};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader};
use std::sync::Mutex;

use bio::bio_types::strand::ReqStrand::{Forward, Reverse};
use bio::io::fasta;
use fishers_exact::fishers_exact;
use itertools::Itertools;
use mathru::statistics::test::{ChiSquare, Test};
use ndarray::{Array3, Axis, stack};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rust_htslib::{bam, bam::{ext::BamRecordExtensions, IndexedReader, Read}};
use rust_htslib::bam::record::Aux;
use rust_lapper::{Interval, Lapper};
use seq_io::fasta::{Reader, Record};

use crate::feature::Feature;

#[derive(Default, Clone, Debug)]
pub struct Region {
    pub(crate) chr: String,
    pub(crate) start: u32,
    // 1-based, inclusive
    pub(crate) end: u32,
    // 1-based, exclusive
    pub(crate) gene_id: Option<String>,
    // if load annotation, this field will tell which gene this region covers. Multiple gene separated by comma
}

impl Region {
    pub fn new(region: String) -> Region {
        // region format: chr:start-end
        if !region.contains(":") {
            let chr = region;
            return Region {
                chr,
                start: 0,
                end: 0,
                gene_id: None,
            };
        } else if region.contains(":") && region.contains("-") {
            let region_vec: Vec<&str> = region.split(":").collect();
            let chr = region_vec[0].to_string();
            let pos_vec: Vec<&str> = region_vec[1].split("-").collect();
            let start = pos_vec[0].parse::<u32>().unwrap();
            let end = pos_vec[1].parse::<u32>().unwrap();
            let gene_id = None;
            assert!(start <= end);
            return Region {
                chr,
                start,
                end,
                gene_id,
            };
        } else {
            panic!("region format error!");
        }
    }
    pub fn to_string(&self) -> String {
        return format!("{}:{}-{}", self.chr, self.start, self.end);
    }
}

#[derive(Default, Debug, Clone)]
pub struct BaseQual {
    pub a: Vec<i32>,
    pub c: Vec<i32>,
    pub g: Vec<i32>,
    pub t: Vec<i32>,
}

#[derive(Default, Debug, Clone)]
pub struct MapQual {
    pub a: Vec<i32>,
    pub c: Vec<i32>,
    pub g: Vec<i32>,
    pub t: Vec<i32>,
}

#[derive(Default, Debug, Clone)]
pub struct BaseStrands {
    pub a: [i32; 2],
    // [forward, backward]
    pub c: [i32; 2],
    pub g: [i32; 2],
    pub t: [i32; 2],
}

#[derive(Default, Debug, Clone)]
pub struct BaseTsStrands {
    pub a: [i32; 2],
    // [forward, backward]
    pub c: [i32; 2],
    pub g: [i32; 2],
    pub t: [i32; 2],
}

#[derive(Default, Debug, Clone)]
pub struct DistanceToEnd {
    pub a: Vec<i64>,
    // allele A to the end of read for every read
    pub c: Vec<i64>,
    // allele C to the end of read for every read
    pub g: Vec<i64>,
    // allele G to the end of read for every read
    pub t: Vec<i64>,
    // allele T to the end of read for every read
}


#[derive(Default, Debug, Clone)]
pub struct BaseFreq {
    pub a: u32,
    pub c: u32,
    pub g: u32,
    pub t: u32,
    pub n: u32,
    // number of introns
    pub d: u32,
    // number of deletions
    pub ni: u32,
    // number of insertions
    pub i: bool,
    // whether this position falls in an insertion
    pub ref_pos: usize,
    // 0-based, position on reference
    pub ref_base: char,
    // reference base, A,C,G,T,N,-
    pub intron: bool,
    // whether this position is an intron
    pub forward_cnt: u32,
    // number of forward reads covering this position, excluding intron
    pub backward_cnt: u32,
    // number of backward reads covering this position, excluding intron
    pub baseq: BaseQual,
    pub mapq: MapQual,
    pub base_strands: BaseStrands,
    pub base_ts_strands: BaseTsStrands,
    pub transcript_strands: [i32; 2],
    // [forward, backward]
    pub distance_to_end: DistanceToEnd,
}

impl BaseFreq {
    pub fn to_feature(&self) -> Feature {
        let mut feature = Feature::default();
        feature.base_cnt[0] = self.a;
        feature.base_cnt[1] = self.c;
        feature.base_cnt[2] = self.g;
        feature.base_cnt[3] = self.t;
        feature.base_cnt[4] = self.n;
        feature.base_cnt[5] = self.d;
        feature.base_cnt[6] = self.ni;
        match self.ref_base {
            'A' => {
                feature.ref_base = 1;
            }
            'C' => {
                feature.ref_base = 2;
            }
            'G' => {
                feature.ref_base = 3;
            }
            'T' => {
                feature.ref_base = 4;
            }
            'N' => {
                feature.ref_base = 5;
            }
            _ => {
                feature.ref_base = 0;
            }
        }
        feature.avg_baseq[0] = if self.baseq.a.len() > 0 { self.baseq.a.iter().sum::<i32>() as f32 / self.baseq.a.len() as f32 } else { 0.0 };
        feature.avg_baseq[1] = if self.baseq.c.len() > 0 { self.baseq.c.iter().sum::<i32>() as f32 / self.baseq.c.len() as f32 } else { 0.0 };
        feature.avg_baseq[2] = if self.baseq.g.len() > 0 { self.baseq.g.iter().sum::<i32>() as f32 / self.baseq.g.len() as f32 } else { 0.0 };
        feature.avg_baseq[3] = if self.baseq.t.len() > 0 { self.baseq.t.iter().sum::<i32>() as f32 / self.baseq.t.len() as f32 } else { 0.0 };
        feature.avg_mapq[0] = if self.mapq.a.len() > 0 { self.mapq.a.iter().sum::<i32>() as f32 / self.mapq.a.len() as f32 } else { 0.0 };
        feature.avg_mapq[1] = if self.mapq.c.len() > 0 { self.mapq.c.iter().sum::<i32>() as f32 / self.mapq.c.len() as f32 } else { 0.0 };
        feature.avg_mapq[2] = if self.mapq.g.len() > 0 { self.mapq.g.iter().sum::<i32>() as f32 / self.mapq.g.len() as f32 } else { 0.0 };
        feature.avg_mapq[3] = if self.mapq.t.len() > 0 { self.mapq.t.iter().sum::<i32>() as f32 / self.mapq.t.len() as f32 } else { 0.0 };
        feature.forward_cnt[0] = self.base_strands.a[0] as u32;
        feature.forward_cnt[1] = self.base_strands.c[0] as u32;
        feature.forward_cnt[2] = self.base_strands.g[0] as u32;
        feature.forward_cnt[3] = self.base_strands.t[0] as u32;
        feature.reverse_cnt[0] = self.base_strands.a[1] as u32;
        feature.reverse_cnt[1] = self.base_strands.c[1] as u32;
        feature.reverse_cnt[2] = self.base_strands.g[1] as u32;
        feature.reverse_cnt[3] = self.base_strands.t[1] as u32;
        feature.ts_forward_cnt = (self.base_ts_strands.a[0] + self.base_ts_strands.c[0] + self.base_ts_strands.g[0] + self.base_ts_strands.t[0]) as u32;
        feature.ts_reverse_cnt = (self.base_ts_strands.a[1] + self.base_ts_strands.c[1] + self.base_ts_strands.g[1] + self.base_ts_strands.t[1]) as u32;
        // feature.ts_forward_cnt[0] = self.base_ts_strands.a[0] as u32;
        // feature.ts_forward_cnt[1] = self.base_ts_strands.c[0] as u32;
        // feature.ts_forward_cnt[2] = self.base_ts_strands.g[0] as u32;
        // feature.ts_forward_cnt[3] = self.base_ts_strands.t[0] as u32;
        // feature.ts_reverse_cnt[0] = self.base_ts_strands.a[1] as u32;
        // feature.ts_reverse_cnt[1] = self.base_ts_strands.c[1] as u32;
        // feature.ts_reverse_cnt[2] = self.base_ts_strands.g[1] as u32;
        // feature.ts_reverse_cnt[3] = self.base_ts_strands.t[1] as u32;
        return feature;
    }
    pub fn subtract(&mut self, base: u8) {
        match base {
            b'A' => {
                assert!(self.a > 0);
                self.a -= 1;
            }
            b'C' => {
                assert!(self.c > 0);
                self.c -= 1;
            }
            b'G' => {
                assert!(self.g > 0);
                self.g -= 1;
            }
            b'T' => {
                assert!(self.t > 0);
                self.t -= 1;
            }
            b'N' => {
                assert!(self.n > 0);
                self.n -= 1;
            }
            b'-' => {
                assert!(self.d > 0);
                self.d -= 1;
            }
            b'*' => {
                return;
            }
            _ => {
                panic!("Invalid base: {}", base as char);
            }
        }
    }

    pub fn add(&mut self, base: u8) {
        match base {
            b'A' => {
                self.a += 1;
            }
            b'C' => {
                self.c += 1;
            }
            b'G' => {
                self.g += 1;
            }
            b'T' => {
                self.t += 1;
            }
            b'N' => {
                self.n += 1;
            }
            b'-' => {
                self.d += 1;
            }
            b'*' => {
                return;
            }
            _ => {
                panic!("Invalid base: {}", base as char);
            }
        }
    }

    pub fn get_depth_include_intron(&self) -> u32 {
        self.a + self.c + self.g + self.t + self.d + self.n
    }

    pub fn get_depth_exclude_intron_deletion(&self) -> u32 {
        self.a + self.c + self.g + self.t
    }

    pub fn get_two_major_alleles(&self, ref_base: char) -> (char, u32, char, u32) {
        let mut x: Vec<(char, u32)> = [('A', self.a), ('C', self.c), ('G', self.g), ('T', self.t)].iter().cloned().collect();
        // sort by count: u32
        x.sort_by(|a, b| b.1.cmp(&a.1));
        if x[0].0 != ref_base && x[1].0 != ref_base {
            if x[2].0 == ref_base && x[1].1 == x[2].1 {
                return (x[0].0, x[0].1, x[2].0, x[2].1);
            } else if x[3].0 == ref_base && x[1].1 == x[3].1 {
                return (x[0].0, x[0].1, x[3].0, x[3].1);
            } else {
                return (x[0].0, x[0].1, x[1].0, x[1].1);
            }
        } else {
            return (x[0].0, x[0].1, x[1].0, x[1].1);
        }
    }

    pub fn get_none_ref_count(&self) -> u32 {
        match self.ref_base {
            'A' => self.c + self.g + self.t + self.d,
            'C' => self.a + self.g + self.t + self.d,
            'G' => self.a + self.c + self.t + self.d,
            'T' => self.a + self.c + self.g + self.d,
            _ => {
                0
            }
        }
    }

    pub fn get_alt_allele_count(&self) -> u32 {
        match self.ref_base {
            'A' => self.c + self.g + self.t,
            'C' => self.a + self.g + self.t,
            'G' => self.a + self.c + self.t,
            'T' => self.a + self.c + self.g,
            _ => {
                0
            }
        }
    }
}


#[derive(Default, Debug, Clone)]
pub struct Pixel {
    pub ref_base: u8,   // A:1, C: 2, G: 3, T: 4, N: 5
    pub base: u8,  // A:1, C: 2, G: 3, T: 4, N: 5, Del: 6
    pub insertion: u8, // 0: no insertion, 1: insertion
    pub baseq: u8,
    pub strand: u8,    // +: 1, -: 2
    pub ts_strand: u8, // +: 1, -: 2
    pub mapq: u8,
}

impl Pixel {
    pub fn to_vec(&self) -> Vec<u8> {
        return vec![self.ref_base, self.base, self.insertion, self.baseq, self.strand, self.ts_strand, self.mapq];
    }
}

#[derive(Default, Debug, Clone)]
pub struct ImageTensor {
    pub tensor: Vec<Vec<Pixel>>,
    pub allele_cnts: Vec<[i32; 6]>,  // A, C, G, T, N, Del
    pub ref_bases: Vec<u8>,
}

impl ImageTensor {
    pub fn get_alt_fraction_depth(&self, col: usize, min_baseq: u8) -> (f32, i32, i32) {
        let mut depth = 0;
        let mut alt_allele_cnt = 0;
        for row in self.tensor.iter() {
            let pixel = &row[col];
            if pixel.base != pixel.ref_base && pixel.ref_base != 5 && pixel.base != 5 && pixel.base != 0 && pixel.base != 6 && pixel.baseq >= min_baseq {
                alt_allele_cnt += 1;
            }
            if pixel.base != 5 && pixel.base != 0 && pixel.base != 6 && pixel.baseq >= min_baseq {
                depth += 1;
            }
        }
        if depth == 0 {
            return (0.0, 0, 0);
        } else {
            return (alt_allele_cnt as f32 / depth as f32, alt_allele_cnt, depth);
        }
    }

    pub fn get_alt_fraction_depth2(&self, col: usize) -> (f32, i32, i32) {
        let ref_base = self.ref_bases[col];
        let ac_vec = self.allele_cnts[col];
        let depth = ac_vec[0] + ac_vec[1] + ac_vec[2] + ac_vec[3];
        let alt_allele_cnt = match ref_base {
            1 => ac_vec[1] + ac_vec[2] + ac_vec[3], // A
            2 => ac_vec[0] + ac_vec[2] + ac_vec[3], // C
            3 => ac_vec[0] + ac_vec[1] + ac_vec[3], // G
            4 => ac_vec[0] + ac_vec[1] + ac_vec[2], // T
            _ => 0,
        };
        if depth == 0 {
            return (0.0, 0, 0);
        } else {
            return (alt_allele_cnt as f32 / depth as f32, alt_allele_cnt, depth);
        }
    }

    pub fn get_col(&self, col: usize) -> Vec<Pixel> {
        let mut col_vec: Vec<Pixel> = Vec::new();
        for row in self.tensor.iter() {
            col_vec.push(row[col].clone());
        }
        return col_vec;
    }

    pub fn get_feature_maxrix(&self, col: usize, flank_size: u32) -> Array3<u8> {
        let mut start = col as i64 - flank_size as i64;
        let end = col as i64 + flank_size as i64;
        let row_size = self.tensor.len();
        let col_size = self.tensor[0].len();
        let mut matrix: Vec<Vec<Vec<u8>>> = Vec::new(); // col_size, row_size, feature_dim
        // println!("start: {}, end: {}", start, end);
        for i in start..=end {
            if i < 0 || i >= col_size as i64 {
                let mut column: Vec<Vec<u8>> = Vec::new();
                for _ in 0..row_size {
                    column.push(Pixel::default().to_vec());
                }
                matrix.push(column);
            } else {
                let mut column: Vec<Vec<u8>> = Vec::new();
                for pixel in self.get_col(i as usize) {
                    column.push(pixel.to_vec());
                }
                matrix.push(column);
            }
        }
        let dim0 = matrix.len();    // 2*flank_size + 1
        let dim1 = matrix[0].len(); // Depth
        let dim2 = matrix[0][0].len();  // Channels
        let flattened: Vec<u8> = matrix.into_iter().flatten().flatten().collect();
        let array = Array3::from_shape_vec((dim0, dim1, dim2), flattened).unwrap(); // L, W, C
        // // remove all zero (padding) rows or all five (intron) rows
        // let mut selected_indices: Vec<usize> = Vec::new();
        // for i in 0..dim1 {
        //     let sum: i32 = array.slice(s![.., i, 1]).iter().map(|&x| x as i32).sum();
        //     if sum != 0 && sum != 5 * dim0 as i32 {
        //         selected_indices.push(i);
        //     }
        // }

        // remove center base allele not in {A,C,G,T,D}
        let mut selected_indices: Vec<usize> = Vec::new();
        for i in 0..dim1 {
            let base = array[[dim0 / 2, i, 1]];
            if base != 5 && base != 0 {
                selected_indices.push(i);
            }
        }

        // sort selected_indices by base allele at the center, decreasing order: I, D, N, T, G, C, A, padding
        selected_indices.sort_by(|a, b| {
            let a_base = array[[dim0 / 2, *a, 1]];
            let b_base = array[[dim0 / 2, *b, 1]];
            b_base.cmp(&a_base)
        });

        // Create an empty vector to hold the selected sub-arrays
        let mut selected_subarrays = Vec::new();
        // Iterate over the indices and collect the sub-arrays
        for &index in &selected_indices {
            selected_subarrays.push(array.index_axis(Axis(1), index).to_owned());
        }
        // Stack the selected subarrays along the second axis
        let selected_subarrays = stack(Axis(1), &selected_subarrays.iter().map(|a| a.view()).collect::<Vec<_>>()).expect("Stacking failed");
        return selected_subarrays;
    }

    pub fn get_feature_maxrix2(&self, col: usize, flank_size: u32, max_depth: u32) -> Array3<u8> {
        let start = col as i64 - flank_size as i64;
        let end = col as i64 + flank_size as i64;
        let row_size = self.tensor.len();
        let row_indices: Vec<usize> = if row_size <= max_depth as usize {
            (0..row_size).collect()
        } else {
            let mut rng = rand::thread_rng();
            (0..row_size).collect::<Vec<_>>().choose_multiple(&mut rng, max_depth as usize).cloned().collect()
        };
        let col_size = self.tensor[0].len();
        let mut matrix: Vec<Vec<Vec<u8>>> = Vec::new(); // col_size(2*flank_size + 1), row_size(depth), feature_dim(channels)
        // println!("start: {}, end: {}", start, end);
        for i in start..=end {
            if i < 0 || i >= col_size as i64 {
                let mut column: Vec<Vec<u8>> = Vec::new();
                for _ in &row_indices {
                    column.push(Pixel::default().to_vec());
                }
                matrix.push(column);
            } else {
                let mut column: Vec<Vec<u8>> = Vec::new();
                // apply row indices
                for &row_idx in &row_indices {
                    let pixel = self.tensor[row_idx][i as usize].clone();
                    column.push(pixel.to_vec());
                }
                matrix.push(column);
            }
        }
        let dim0 = matrix.len();    // 2*flank_size + 1
        let dim1 = matrix[0].len(); // Depth
        let dim2 = matrix[0][0].len();  // Channels
        let flattened: Vec<u8> = matrix.into_iter().flatten().flatten().collect();
        let array = Array3::from_shape_vec((dim0, dim1, dim2), flattened).unwrap(); // L, W, C
        // // remove all zero (padding) rows or all five (intron) rows
        // let mut selected_indices: Vec<usize> = Vec::new();
        // for i in 0..dim1 {
        //     let sum: i32 = array.slice(s![.., i, 1]).iter().map(|&x| x as i32).sum();
        //     if sum != 0 && sum != 5 * dim0 as i32 {
        //         selected_indices.push(i);
        //     }
        // }
        // remove center base allele not in {A,C,G,T,D}
        let mut selected_indices: Vec<usize> = Vec::new();
        for i in 0..dim1 {
            let base = array[[dim0 / 2, i, 1]];
            if base != 5 && base != 0 {
                selected_indices.push(i);
            }
        }

        if selected_indices.is_empty() {
            // Return an empty Array3 with the appropriate shape
            return Array3::zeros((dim0, 0, dim2));
        }

        // Randomly sample selected_indices if larger than 50000
        // if selected_indices.len() > max_depth as usize {
        //     let mut rng = rand::thread_rng();
        //     selected_indices = selected_indices.choose_multiple(&mut rng, max_depth as usize).cloned().collect();
        //     println!("Randomly sample {} reads", max_depth);
        // }

        // sort selected_indices by base allele at the center, decreasing order: I, D, N, T, G, C, A, padding
        selected_indices.sort_by(|a, b| {
            let a_base = array[[dim0 / 2, *a, 1]];
            let b_base = array[[dim0 / 2, *b, 1]];
            b_base.cmp(&a_base)
        });

        // Create an empty vector to hold the selected sub-arrays
        let mut selected_subarrays = Vec::new();
        // Iterate over the indices and collect the sub-arrays
        for &index in &selected_indices {
            selected_subarrays.push(array.index_axis(Axis(1), index).to_owned());
        }
        // Stack the selected subarrays along the second axis
        let selected_subarrays = stack(Axis(1), &selected_subarrays.iter().map(|a| a.view()).collect::<Vec<_>>()).expect("Stacking failed");
        return selected_subarrays;
    }
}

pub fn find_covering_region_indices(regions: &[(usize, usize)], position: usize) -> Vec<usize> {
    // Use binary search to find the first region that might cover the position
    let start_index = regions.binary_search_by(|&(_, end)| {
        if end <= position {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    }).unwrap_or_else(|x| x);

    // Collect indices of all regions that cover the position
    (start_index..regions.len())
        .take_while(|&i| regions[i].0 <= position)
        .filter(|&i| regions[i].0 <= position && position < regions[i].1)
        .collect()
}

pub fn load_reference(ref_path: String) -> HashMap<String, Vec<u8>> {
    let mut ref_seqs: HashMap<String, Vec<u8>> = HashMap::new();
    let reader = fasta::Reader::from_file(ref_path).unwrap();
    for r in reader.records() {
        let ref_record = r.unwrap();
        ref_seqs.insert(ref_record.id().to_string(), ref_record.seq().to_vec());
    }
    return ref_seqs;
}


pub fn parse_fai(fai_path: &str) -> Vec<(String, u32)> {
    let mut contig_lengths: Vec<(String, u32)> = Vec::new();
    let file = File::open(fai_path).unwrap();
    let reader = BufReader::new(file);
    for r in reader.lines() {
        let line = r.unwrap().clone();
        let parts: Vec<&str> = line.split('\t').collect();
        contig_lengths.push((parts[0].to_string(), parts[1].parse().unwrap()));
    }
    return contig_lengths;
}

pub fn find_isolated_regions_with_depth(bam_path: &str, chr: &str, ref_len: u32, min_mapq: u8, min_read_length: usize) -> Vec<Region> {
    let mut isolated_regions: Vec<Region> = Vec::new();
    let mut depth_vec: Vec<u32> = vec![0; ref_len as usize];
    let mut bam: bam::IndexedReader = bam::IndexedReader::from_path(bam_path).unwrap();
    let header = bam.header().clone();
    bam.fetch(chr).unwrap();
    for r in bam.records() {
        let record = r.unwrap();
        if record.mapq() < min_mapq || record.seq_len() < min_read_length || record.is_unmapped() || record.is_secondary() || record.is_supplementary() {
            continue;
        }
        let ref_start = record.reference_start();   // 0-based, left-closed
        let ref_end = record.reference_end();   // 0-based, right-open
        for i in ref_start..ref_end {
            depth_vec[i as usize] += 1;
        }
    }
    let mut region_start = -1;
    let mut region_end = -1;
    for i in 0..ref_len {
        if depth_vec[i as usize] == 0 {
            if region_end > region_start {
                assert!(region_start >= 0);
                assert!(region_end >= 0);
                isolated_regions.push(Region { chr: chr.to_string(), start: (region_start + 1) as u32, end: (region_end + 2) as u32, gene_id: None });
                region_start = -1;
                region_end = -1;
            }
        } else {
            if region_start == -1 {
                region_start = i as i32;
                region_end = i as i32;
            } else {
                region_end = i as i32;
            }
        }
    }
    if region_end > region_start {
        isolated_regions.push(Region { chr: chr.to_string(), start: (region_start + 1) as u32, end: (region_end + 2) as u32, gene_id: None });
        region_start = -1;
        region_end = -1;
    }
    return isolated_regions;
}

pub fn parse_bed(bed_path: String) -> HashMap<String, VecDeque<Region>> {
    let mut target_regions: HashMap<String, VecDeque<Region>> = HashMap::new();    // key is chr, value is a stack of regions
    let file = File::open(bed_path).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.unwrap();
        let parts = line.split('\t').collect::<Vec<&str>>();
        let chr = parts[0].to_string();
        let start = parts[1].parse::<u32>().unwrap();   // bed file is 0-based, left inclusive, right exclusive
        let end = parts[2].parse::<u32>().unwrap();
        if !target_regions.contains_key(&chr) {
            target_regions.insert(chr.clone(), VecDeque::new());
        }
        target_regions.get_mut(&chr).unwrap().push_back(Region { chr: chr.clone(), start: start + 1, end: end + 1, gene_id: None });
    }
    return target_regions;
}

pub fn parse_annotation(anno_path: String) -> (HashMap<String, VecDeque<Region>>, HashMap<String, Vec<Interval<usize, u8>>>) {
    let mut gene_regions: HashMap<String, VecDeque<Region>> = HashMap::new();    // key is chr, value is a stack of gene regions
    let mut exon_regions: HashMap<String, Vec<Interval<usize, u8>>> = HashMap::new();  // key is gene id, value is exon regions
    let file = File::open(anno_path).unwrap();
    let reader = BufReader::new(file);
    let mut invs: Vec<Interval<usize, u8>> = Vec::new(); // for merging gene exons
    let mut gene_id: String = String::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with("#") { continue; }
        let parts = line.split('\t').collect::<Vec<&str>>();
        let seqname = parts[0].to_string();
        let feature = parts[2];
        let start = parts[3].parse::<u32>().unwrap();   // 1-based, inclusive
        let end = parts[4].parse::<u32>().unwrap(); // 1-based, inclusive
        if feature == "gene" {
            if invs.len() > 0 {
                exon_regions.insert(gene_id.clone(), invs.clone());
                invs.clear();
            }
            if !gene_regions.contains_key(&seqname) {
                gene_regions.insert(seqname.clone(), VecDeque::new());
            }
            for subpart in parts[8].trim_end().split(";").collect::<Vec<&str>>() {
                if subpart.starts_with("gene_id") {
                    gene_id = subpart.replace("gene_id=", "");  // gff3 format
                    break;
                }
                if subpart.starts_with("gene_id") {
                    gene_id = subpart.replace("gene_id ", "").replace("\"", "");  // gtf format
                    break;
                }
            }
            let mut top = gene_regions.get_mut(&seqname).unwrap().pop_back();
            if top.is_some() {
                let mut top = top.unwrap();
                assert!(start >= top.start, "Error: annotation file is not sorted. {}:{}-{}", seqname, start, end);
                // if overlap, merge the overlapped regions
                if top.end <= start {
                    // end of top region is exclusive, so top.end == start is not overlap
                    gene_regions.get_mut(&seqname).unwrap().push_back(top);
                    gene_regions.get_mut(&seqname).unwrap().push_back(Region { chr: seqname.clone(), start: start, end: end + 1, gene_id: Option::from(gene_id.clone()) });
                } else if top.end < end + 1 {
                    // top.end is exclusive, end is inclusive
                    // merge two overlapped regions
                    top.end = end + 1;
                    top.gene_id = Option::from(top.gene_id.unwrap() + "," + &gene_id);
                    gene_regions.get_mut(&seqname).unwrap().push_back(top);
                } else {
                    // equal end or contained
                    top.gene_id = Option::from(top.gene_id.unwrap() + "," + &gene_id);
                    gene_regions.get_mut(&seqname).unwrap().push_back(top);
                }
            } else {
                // first gene region in stack
                gene_regions.get_mut(&seqname).unwrap().push_back(Region { chr: seqname.clone(), start: start, end: end + 1, gene_id: Option::from(gene_id.clone()) });
            }
        } else if feature == "CDS" {
            let mut exon_gene_id = String::new();
            for subpart in parts[8].trim_end().split(";").collect::<Vec<&str>>() {
                if subpart.starts_with("gene_id") {
                    exon_gene_id = subpart.replace("gene_id=", "");  // gff3 format
                    break;
                }
                if subpart.starts_with("gene_id") {
                    exon_gene_id = subpart.replace("gene_id ", "").replace("\"", "");  // gtf format
                    break;
                }
            }
            assert!(exon_gene_id == gene_id, "Error: gene_id in gene and exon are different: gene_id:{}, exon_gene_id:{}", gene_id, exon_gene_id);
            invs.push(Interval { start: start as usize, stop: (end + 1) as usize, val: 0 });    // 1-based,
        } else {
            continue;
        }
    }
    if invs.len() > 0 {
        exon_regions.insert(gene_id.clone(), invs.clone());
        invs.clear();
    }
    return (gene_regions, exon_regions);
}

pub fn lapper_intervals(query_regions: &Vec<Region>, target_regions: &VecDeque<Region>) -> Vec<Region> {
    let mut result_regions = Vec::new();
    let mut invs: Vec<Interval<usize, usize>> = Vec::new();
    for region in target_regions {
        invs.push(Interval { start: region.start as usize, stop: region.end as usize, val: 0 });
    }
    let mut interval_tree = Lapper::new(invs);
    for q in query_regions {
        let q_inv = Interval { start: q.start as usize, stop: q.end as usize, val: 0 };
        for h_inv in interval_tree.find(q.start as usize, q.end as usize) {
            let intersected_start = q_inv.start.max(h_inv.start);
            let intersected_end = q_inv.stop.min(h_inv.stop);
            assert!(intersected_start < intersected_end, "Error: intersected_start >= intersected_end, query:{:?}", q_inv);
            result_regions.push(Region { chr: q.chr.clone(), start: intersected_start as u32, end: intersected_end as u32, gene_id: None });
        }
    }
    return result_regions;
}

pub fn intersect_interested_regions(alignment_regions: &Vec<Region>, interested_regions: &HashMap<String, VecDeque<Region>>, thread_size: usize) -> Vec<Region> {
    let intersected_regions: Mutex<Vec<Region>> = Mutex::new(Vec::new());
    let mut region_map = HashMap::new();
    for region in alignment_regions {
        if !region_map.contains_key(&region.chr) {
            region_map.insert(region.chr.clone(), Vec::new());
        }
        region_map.get_mut(&region.chr).unwrap().push(region.clone());
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_size - 1).build().unwrap();
    let contig_names = region_map.keys().collect::<Vec<&String>>();
    pool.install(|| {
        contig_names.par_iter().for_each(|ctg| {
            if interested_regions.contains_key(*ctg) {
                let chr_intersected_region = lapper_intervals(region_map.get(*ctg).unwrap(), interested_regions.get(*ctg).unwrap());
                for region in chr_intersected_region {
                    intersected_regions.lock().unwrap().push(region);
                }
            }
        });
    });

    return intersected_regions.into_inner().unwrap();
}

pub fn multithread_produce3(bam_file: String, ref_file: String, thread_size: usize, contigs: Option<Vec<String>>, min_mapq: u8, min_read_length: usize) -> Vec<Region> {
    let results: Mutex<Vec<Region>> = Mutex::new(Vec::new());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_size - 1).build().unwrap();
    let bam = bam::IndexedReader::from_path(bam_file.clone()).unwrap();
    let bam_header = bam.header().clone();
    let fai_path = ref_file + ".fai";
    if fs::metadata(&fai_path).is_err() {
        panic!("Reference index file .fai does not exist.");
    }
    let contig_lengths = parse_fai(fai_path.as_str());
    let mut contig_names: VecDeque<String> = VecDeque::new();
    if contigs.is_some() {
        for ctg in contigs.unwrap().iter() {
            contig_names.push_back(ctg.clone());
        }
    } else {
        // for ctg in bam_header.target_names() {
        //     contig_names.push_back(std::str::from_utf8(ctg).unwrap().to_string().clone());
        // }
        for (ctg, _) in contig_lengths.iter() {
            contig_names.push_back(ctg.clone());
        }
    }
    pool.install(|| {
        contig_names.par_iter().for_each(|ctg| {
            let isolated_regions = find_isolated_regions_with_depth(bam_file.as_str(), ctg, contig_lengths.iter().find(|(chr, _)| chr == ctg).unwrap().1, min_mapq, min_read_length);
            for region in isolated_regions {
                results.lock().unwrap().push(region);
            }
        });
    });
    return results.into_inner().unwrap().clone();
}

pub fn read_references(ref_path: &str) -> HashMap<String, Vec<u8>> {
    let mut references: HashMap<String, Vec<u8>> = HashMap::new();
    let mut reader = Reader::from_path(ref_path).unwrap();
    while let Some(record) = reader.next() {
        let record = record.expect("Error reading record");
        references.insert(record.id().unwrap().to_string(), record.full_seq().to_vec());
    }
    return references;
}

#[derive(Default, Debug, Clone)]
pub struct Profile {
    pub freq_vec: Vec<BaseFreq>,
    pub image: Vec<Pixel>,
    pub region: Region,
}

impl Profile {
    pub fn init_with_pileup(&mut self, bam_path: &str, region: &Region, ref_seq: &Vec<u8>, min_mapq: u8, min_baseq: u8, min_read_length: usize) {
        // When region is large and the number of reads is large, the runtime of init_profile_with_pileup is time-consuming.
        // This function is used to fill the profile by parsing each read in the bam file instead of using pileup.
        let mut bam: bam::IndexedReader = bam::IndexedReader::from_path(bam_path).unwrap();
        bam.fetch((region.chr.as_str(), region.start, region.end)).unwrap();
        let vec_size = (region.end - region.start) as usize;    // end is exclusive
        self.freq_vec = vec![BaseFreq::default(); vec_size];
        self.region = region.clone();
        let freq_vec_pos = region.start as usize - 1;    // the first position on reference, 0-based, inclusive

        // fill the ref_base field in each BaseFreq
        for i in 0..vec_size {
            self.freq_vec[i].ref_base = ref_seq[freq_vec_pos + i] as char;
            self.freq_vec[i].ref_pos = freq_vec_pos + i;
        }

        for r in bam.records() {
            let record = r.unwrap();
            if record.mapq() < min_mapq || record.seq_len() < min_read_length || record.is_unmapped() || record.is_secondary() || record.is_supplementary() {
                continue;
            }
            let mapq = record.mapq() as i32;
            let qname = std::str::from_utf8(record.qname()).unwrap().to_string();
            let seq = record.seq();
            let base_qual = record.qual();
            let strand = if record.strand() == Forward { 0 } else { 1 };
            let mut ts = Aux::Char(b'*');
            match record.aux(b"ts") {
                Ok(value) => {
                    ts = value;
                }
                Err(_) => {}
            }
            let start_pos = record.pos() as usize;  // 0-based
            let cigar = record.cigar();
            let leading_softclips = cigar.leading_softclips();
            let trailing_softclips = cigar.trailing_softclips();

            let mut pos_in_freq_vec: i32 = start_pos as i32 - freq_vec_pos as i32;
            let mut pos_in_read = if leading_softclips > 0 { leading_softclips as usize } else { 0 };
            let cigars = cigar.to_vec();
            for cg_idx in 0..cigars.len() {
                let cg = cigars[cg_idx];
                match cg.char() as u8 {
                    b'S' | b'H' => {
                        continue;
                    }
                    b'M' | b'X' | b'=' => {
                        for cgi in 0..cg.len() {
                            if pos_in_freq_vec < 0 {
                                pos_in_freq_vec += 1;
                                pos_in_read += 1;
                                continue;
                            }
                            if pos_in_freq_vec >= vec_size as i32 {
                                break;
                            }
                            let base = seq[pos_in_read] as char;
                            let mut baseq = base_qual[pos_in_read] as i32;
                            // baseq = if baseq < 30 { baseq } else { 30 };


                            // close to left read end or right read end, check whether current position is in polyA tail
                            let ref_base = self.freq_vec[pos_in_freq_vec as usize].ref_base;

                            // calculate distance to read end of each allele, for filtering variants that the average distance of each allele is significantly different
                            let mut dist = 0;
                            if (pos_in_read as i64 - leading_softclips).abs() < (pos_in_read as i64 - (seq.len() as i64 - trailing_softclips)).abs() {
                                dist = pos_in_read as i64 - leading_softclips;    // positive value
                            } else {
                                dist = pos_in_read as i64 - (seq.len() as i64 - trailing_softclips);    // negative value
                            }

                            let mut rts = Forward;
                            if strand == 0 {
                                if ts == Aux::Char(b'+') {
                                    self.freq_vec[pos_in_freq_vec as usize].transcript_strands[0] += 1; // read +, ts +, transcript +
                                    rts = Forward;
                                } else if ts == Aux::Char(b'-') {
                                    self.freq_vec[pos_in_freq_vec as usize].transcript_strands[1] += 1; // read +, ts -, transcript -
                                    rts = Reverse;
                                }
                            } else if strand == 1 {
                                if ts == Aux::Char(b'+') {
                                    self.freq_vec[pos_in_freq_vec as usize].transcript_strands[1] += 1; // read -, ts +, transcript -
                                    rts = Reverse;
                                } else if ts == Aux::Char(b'-') {
                                    self.freq_vec[pos_in_freq_vec as usize].transcript_strands[0] += 1; // read -, ts -, transcript +
                                    rts = Forward;
                                }
                            }

                            match base {
                                'A' | 'a' => {
                                    self.freq_vec[pos_in_freq_vec as usize].a += 1;
                                    self.freq_vec[pos_in_freq_vec as usize].baseq.a.push(baseq);
                                    self.freq_vec[pos_in_freq_vec as usize].mapq.a.push(mapq);
                                    if strand == 0 {
                                        self.freq_vec[pos_in_freq_vec as usize].base_strands.a[0] += 1;
                                    } else {
                                        self.freq_vec[pos_in_freq_vec as usize].base_strands.a[1] += 1;
                                    }
                                    if rts == Forward {
                                        self.freq_vec[pos_in_freq_vec as usize].base_ts_strands.a[0] += 1;
                                    } else {
                                        self.freq_vec[pos_in_freq_vec as usize].base_ts_strands.a[1] += 1;
                                    }
                                    self.freq_vec[pos_in_freq_vec as usize].distance_to_end.a.push(dist);
                                }
                                'C' | 'c' => {
                                    self.freq_vec[pos_in_freq_vec as usize].c += 1;
                                    self.freq_vec[pos_in_freq_vec as usize].baseq.c.push(baseq);
                                    self.freq_vec[pos_in_freq_vec as usize].mapq.c.push(mapq);
                                    if strand == 0 {
                                        self.freq_vec[pos_in_freq_vec as usize].base_strands.c[0] += 1;
                                    } else {
                                        self.freq_vec[pos_in_freq_vec as usize].base_strands.c[1] += 1;
                                    }
                                    if rts == Forward {
                                        self.freq_vec[pos_in_freq_vec as usize].base_ts_strands.c[0] += 1;
                                    } else {
                                        self.freq_vec[pos_in_freq_vec as usize].base_ts_strands.c[1] += 1;
                                    }
                                    self.freq_vec[pos_in_freq_vec as usize].distance_to_end.c.push(dist);
                                }
                                'G' | 'g' => {
                                    self.freq_vec[pos_in_freq_vec as usize].g += 1;
                                    self.freq_vec[pos_in_freq_vec as usize].baseq.g.push(baseq);
                                    self.freq_vec[pos_in_freq_vec as usize].mapq.g.push(mapq);
                                    if strand == 0 {
                                        self.freq_vec[pos_in_freq_vec as usize].base_strands.g[0] += 1;
                                    } else {
                                        self.freq_vec[pos_in_freq_vec as usize].base_strands.g[1] += 1;
                                    }
                                    if rts == Forward {
                                        self.freq_vec[pos_in_freq_vec as usize].base_ts_strands.g[0] += 1;
                                    } else {
                                        self.freq_vec[pos_in_freq_vec as usize].base_ts_strands.g[1] += 1;
                                    }
                                    self.freq_vec[pos_in_freq_vec as usize].distance_to_end.g.push(dist);
                                }
                                'T' | 't' => {
                                    self.freq_vec[pos_in_freq_vec as usize].t += 1;
                                    self.freq_vec[pos_in_freq_vec as usize].baseq.t.push(baseq);
                                    self.freq_vec[pos_in_freq_vec as usize].mapq.t.push(mapq);
                                    if strand == 0 {
                                        self.freq_vec[pos_in_freq_vec as usize].base_strands.t[0] += 1;
                                    } else {
                                        self.freq_vec[pos_in_freq_vec as usize].base_strands.t[1] += 1;
                                    }
                                    if rts == Forward {
                                        self.freq_vec[pos_in_freq_vec as usize].base_ts_strands.t[0] += 1;
                                    } else {
                                        self.freq_vec[pos_in_freq_vec as usize].base_ts_strands.t[1] += 1;
                                    }
                                    self.freq_vec[pos_in_freq_vec as usize].distance_to_end.t.push(dist);
                                }
                                _ => {
                                    println!("Invalid nucleotide base: {}", base);
                                }
                            }
                            if strand == 0 {
                                self.freq_vec[pos_in_freq_vec as usize].forward_cnt += 1;
                            } else {
                                self.freq_vec[pos_in_freq_vec as usize].backward_cnt += 1;
                            }

                            pos_in_freq_vec += 1;
                            pos_in_read += 1;
                        }
                    }
                    b'D' => {
                        for _ in 0..cg.len() {
                            if pos_in_freq_vec < 0 {
                                pos_in_freq_vec += 1;
                                continue;
                            }
                            if pos_in_freq_vec >= vec_size as i32 {
                                break;
                            }
                            self.freq_vec[pos_in_freq_vec as usize].d += 1;
                            pos_in_freq_vec += 1;
                        }
                    }
                    b'I' => {
                        if pos_in_freq_vec < 1 {
                            // smaller than 1 instead of 0, because insertion is counted as the previous position
                            pos_in_read += cg.len() as usize;
                            continue;
                        }
                        if pos_in_freq_vec >= vec_size as i32 {
                            break;
                        }
                        self.freq_vec[(pos_in_freq_vec - 1) as usize].ni += 1; // insertion is counted as the previous position
                        pos_in_read += cg.len() as usize;
                    }
                    b'N' => {
                        for _ in 0..cg.len() {
                            if pos_in_freq_vec < 0 {
                                pos_in_freq_vec += 1;
                                continue;
                            }
                            if pos_in_freq_vec >= vec_size as i32 {
                                break;
                            }
                            self.freq_vec[pos_in_freq_vec as usize].n += 1;
                            pos_in_freq_vec += 1;
                        }
                    }
                    _ => {
                        panic!("Error: unknown cigar operation: {}", cg.char());
                    }
                }
            }
        }
    }

    pub fn bam_to_image(&mut self, bam_path: &str, region: &Region, ref_seq: &Vec<u8>, min_mapq: u8) -> ImageTensor {
        let mut bam = IndexedReader::from_path(bam_path).expect("Error opening BAM file");
        let tid = bam.header().tid(region.chr.as_bytes()).unwrap();
        let start: i64 = region.start as i64 - 1;   // 0-based, inclusive
        let end: i64 = region.end as i64 - 1;   // 0-based, exclusive
        bam.fetch((tid, start, end)).expect("Error fetching reads");    // fetch is zero-based. start is inclusive, stop is exclusive.
        let mut image_tensor = ImageTensor::default();
        image_tensor.allele_cnts = vec![[0; 6]; (end - start) as usize];    // A,C,G,T,N,D
        image_tensor.ref_bases = vec![0; (end - start) as usize];   // 0: not set, 1: A, 2: C, 3: G, 4: T, 5: N
        for result in bam.records() {
            let tensor_len = (end - start) as usize;
            let mut tensor_row = vec![Pixel::default(); tensor_len];
            let record = result.expect("Error reading record");
            if record.is_unmapped() || record.is_secondary() || record.is_supplementary() || record.mapq() < min_mapq {
                continue;
            }
            // Get position on reference
            let pos = record.pos() as usize;
            // Get sequence
            let seq = record.seq().as_bytes();
            // Get base qualities
            let quals = record.qual();
            // Get mapping quality
            let mapq = record.mapq();
            // Get strand information
            let strand = if record.is_reverse() { 2 } else { 1 };   // 0: not set, 1: forward, 2: reverse
            let mut ts = Aux::Char(b'*');
            match record.aux(b"ts") {
                Ok(value) => {
                    ts = value;
                }
                Err(_) => {}
            }
            let mut ts_strand = 0;
            match ts {
                Aux::Char(b'+') => {
                    ts_strand = 1;
                }
                Aux::Char(b'-') => {
                    ts_strand = 2;
                }
                _ => {}
            }

            // Parse CIGAR string
            let cigar = record.cigar();
            let leading_softclips = cigar.leading_softclips();
            let mut col = pos as i64 - start;   // col may be negative because read start position may be smaller than region start position
            let mut pos_in_read = if leading_softclips > 0 { leading_softclips as usize } else { 0 };
            // fill the ref_base field in each BaseFreq
            for i in 0..tensor_len {
                // tensor_row[i].ref_base = ref_seq[start as usize + i];
                match ref_seq[start as usize + i] {
                    b'A' | b'a' => {
                        tensor_row[i].ref_base = 1;   // 0: not set, 1: A, 2: C, 3: G, 4: T, 5: N
                        image_tensor.ref_bases[i] = 1;
                    }
                    b'C' | b'c' => {
                        tensor_row[i].ref_base = 2;
                        image_tensor.ref_bases[i] = 2;
                    }
                    b'G' | b'g' => {
                        tensor_row[i].ref_base = 3;
                        image_tensor.ref_bases[i] = 3;
                    }
                    b'T' | b't' => {
                        tensor_row[i].ref_base = 4;
                        image_tensor.ref_bases[i] = 4;
                    }
                    _ => {
                        tensor_row[i].ref_base = 5;
                        image_tensor.ref_bases[i] = 5;
                    }
                }
            }

            let cigars = cigar.to_vec();
            for cg_idx in 0..cigars.len() {
                let cg = cigars[cg_idx];
                match cg.char() as u8 {
                    b'S' | b'H' => {
                        continue;
                    }
                    b'M' | b'X' | b'=' => {
                        for cgi in 0..cg.len() {
                            if col < 0 {
                                col += 1;
                                pos_in_read += 1;
                                continue;
                            }
                            if col >= tensor_len as i64 {
                                break;
                            }
                            assert!(col >= 0 && col < tensor_len as i64, "Error: col out of range: {}", col);
                            let base = seq[pos_in_read] as char;
                            let baseq = quals[pos_in_read];
                            match base {
                                'A' | 'a' => {
                                    tensor_row[col as usize].base = 1;   // 0: not set, 1: A, 2: C, 3: G, 4: T, 5: N, 6: D
                                    image_tensor.allele_cnts[col as usize][0] += 1;
                                }
                                'C' | 'c' => {
                                    tensor_row[col as usize].base = 2;
                                    image_tensor.allele_cnts[col as usize][1] += 1;
                                }
                                'G' | 'g' => {
                                    tensor_row[col as usize].base = 3;
                                    image_tensor.allele_cnts[col as usize][2] += 1;
                                }
                                'T' | 't' => {
                                    tensor_row[col as usize].base = 4;
                                    image_tensor.allele_cnts[col as usize][3] += 1;
                                }
                                _ => {
                                    println!("Invalid nucleotide base: {}", base);
                                }
                            }
                            tensor_row[col as usize].baseq = baseq;
                            tensor_row[col as usize].mapq = mapq;
                            tensor_row[col as usize].strand = strand;
                            tensor_row[col as usize].ts_strand = ts_strand;
                            col += 1;
                            pos_in_read += 1;
                        }
                    }
                    b'D' => {
                        for _ in 0..cg.len() {
                            if col < 0 {
                                col += 1;
                                continue;
                            }
                            if col >= tensor_len as i64 {
                                break;
                            }
                            assert!(col >= 0 && col < tensor_len as i64, "Error: col out of range: {}", col);
                            tensor_row[col as usize].base = 6;
                            image_tensor.allele_cnts[col as usize][5] += 1;
                            col += 1;
                        }
                    }
                    b'I' => {
                        if col < 1 {
                            // smaller than 1 instead of 0, because insertion is counted as the previous position
                            pos_in_read += cg.len() as usize;
                            continue;
                        }
                        if col >= tensor_len as i64 {
                            break;
                        }
                        assert!(col >= 1 && col < tensor_len as i64, "Error: col out of range: {}", col);
                        tensor_row[(col - 1) as usize].insertion = 1;
                        pos_in_read += cg.len() as usize;
                    }
                    b'N' => {
                        for _ in 0..cg.len() {
                            if col < 0 {
                                col += 1;
                                continue;
                            }
                            if col >= tensor_len as i64 {
                                break;
                            }
                            assert!(col >= 0 && col < tensor_len as i64, "Error: col out of range: {}", col);
                            tensor_row[col as usize].base = 5;
                            tensor_row[col as usize].baseq = 0;
                            tensor_row[col as usize].mapq = mapq;
                            tensor_row[col as usize].strand = strand;
                            tensor_row[col as usize].ts_strand = ts_strand;
                            image_tensor.allele_cnts[col as usize][4] += 1;
                            col += 1;
                        }
                    }
                    _ => {
                        panic!("Error: unknown cigar operation: {}", cg.char());
                    }
                }
            }
            image_tensor.tensor.push(tensor_row);
        }
        image_tensor
    }


    pub fn bam_to_image2(&mut self, reads_vec: &Vec<bam::Record>, read_idxes: &HashSet<usize>, region: &Region, ref_seq: &Vec<u8>, min_baseq: u8, min_mapq: u8, max_depth: u32) -> ImageTensor {
        let start: i64 = region.start as i64 - 1;   // 0-based, inclusive
        let end: i64 = region.end as i64 - 1;   // 0-based, exclusive
        let mut image_tensor = ImageTensor::default();
        image_tensor.allele_cnts = vec![[0; 6]; (end - start) as usize];    // A,C,G,T,N,D
        image_tensor.ref_bases = vec![0; (end - start) as usize];   // 0: not set, 1: A, 2: C, 3: G, 4: T, 5: N
        let mut read_idxes_vec = read_idxes.iter().collect_vec();
        if read_idxes_vec.len() > max_depth as usize {
            // Randomly select max_depth reads from the Vec
            println!("Chunk: {}:{}-{}, number of reads: {}, down-sampling depth: {}", region.chr, region.start, region.end, read_idxes_vec.len(), max_depth);
            let mut rng = rand::thread_rng();
            read_idxes_vec = read_idxes_vec.choose_multiple(&mut rng, max_depth as usize).cloned().collect();
        }
        for idx in read_idxes_vec {
            let record = reads_vec.get(*idx).unwrap().clone();
            let tensor_len = (end - start) as usize;
            let mut tensor_row = vec![Pixel::default(); tensor_len];
            if record.is_unmapped() || record.is_secondary() || record.is_supplementary() || record.mapq() < min_mapq {
                continue;
            }
            // Get position on reference
            let pos = record.pos() as usize;
            // Get sequence
            let seq = record.seq().as_bytes();
            // Get base qualities
            let quals = record.qual();
            // Get mapping quality
            let mapq = record.mapq();
            // Get strand information
            let strand = if record.is_reverse() { 2 } else { 1 };   // 0: not set, 1: forward, 2: reverse
            let mut ts = Aux::Char(b'*');
            match record.aux(b"ts") {
                Ok(value) => {
                    ts = value;
                }
                Err(_) => {}
            }
            let mut ts_strand = 0;
            match ts {
                Aux::Char(b'+') => {
                    ts_strand = 1;
                }
                Aux::Char(b'-') => {
                    ts_strand = 2;
                }
                _ => {}
            }

            // Parse CIGAR string
            let cigar = record.cigar();
            let leading_softclips = cigar.leading_softclips();
            let mut col = pos as i64 - start;   // col may be negative because read start position may be smaller than region start position
            let mut pos_in_read = if leading_softclips > 0 { leading_softclips as usize } else { 0 };
            // fill the ref_base field in each BaseFreq
            for i in 0..tensor_len {
                // tensor_row[i].ref_base = ref_seq[start as usize + i];
                match ref_seq[start as usize + i] {
                    b'A' | b'a' => {
                        tensor_row[i].ref_base = 1;   // 0: not set, 1: A, 2: C, 3: G, 4: T, 5: N
                        image_tensor.ref_bases[i] = 1;
                    }
                    b'C' | b'c' => {
                        tensor_row[i].ref_base = 2;
                        image_tensor.ref_bases[i] = 2;
                    }
                    b'G' | b'g' => {
                        tensor_row[i].ref_base = 3;
                        image_tensor.ref_bases[i] = 3;
                    }
                    b'T' | b't' => {
                        tensor_row[i].ref_base = 4;
                        image_tensor.ref_bases[i] = 4;
                    }
                    _ => {
                        tensor_row[i].ref_base = 5;
                        image_tensor.ref_bases[i] = 5;
                    }
                }
            }

            let cigars = cigar.to_vec();
            for cg_idx in 0..cigars.len() {
                let cg = cigars[cg_idx];
                match cg.char() as u8 {
                    b'S' | b'H' => {
                        continue;
                    }
                    b'M' | b'X' | b'=' => {
                        for cgi in 0..cg.len() {
                            if col < 0 {
                                col += 1;
                                pos_in_read += 1;
                                continue;
                            }
                            if col >= tensor_len as i64 {
                                break;
                            }
                            assert!(col >= 0 && col < tensor_len as i64, "Error: col out of range: {}", col);
                            let base = seq[pos_in_read] as char;
                            let baseq = quals[pos_in_read];
                            match base {
                                'A' | 'a' => {
                                    tensor_row[col as usize].base = 1;   // 0: not set, 1: A, 2: C, 3: G, 4: T, 5: N, 6: D
                                    if baseq >= min_baseq {
                                        image_tensor.allele_cnts[col as usize][0] += 1;
                                    }
                                }
                                'C' | 'c' => {
                                    tensor_row[col as usize].base = 2;
                                    if baseq >= min_baseq {
                                        image_tensor.allele_cnts[col as usize][1] += 1;
                                    }
                                }
                                'G' | 'g' => {
                                    tensor_row[col as usize].base = 3;
                                    if baseq >= min_baseq {
                                        image_tensor.allele_cnts[col as usize][2] += 1;
                                    }
                                }
                                'T' | 't' => {
                                    tensor_row[col as usize].base = 4;
                                    if baseq >= min_baseq {
                                        image_tensor.allele_cnts[col as usize][3] += 1;
                                    }
                                }
                                _ => {
                                    println!("Invalid nucleotide base: {}", base);
                                }
                            }
                            tensor_row[col as usize].baseq = baseq;
                            tensor_row[col as usize].mapq = mapq;
                            tensor_row[col as usize].strand = strand;
                            tensor_row[col as usize].ts_strand = ts_strand;
                            col += 1;
                            pos_in_read += 1;
                        }
                    }
                    b'D' => {
                        for _ in 0..cg.len() {
                            if col < 0 {
                                col += 1;
                                continue;
                            }
                            if col >= tensor_len as i64 {
                                break;
                            }
                            assert!(col >= 0 && col < tensor_len as i64, "Error: col out of range: {}", col);
                            tensor_row[col as usize].base = 6;
                            image_tensor.allele_cnts[col as usize][5] += 1;
                            col += 1;
                        }
                    }
                    b'I' => {
                        if col < 1 {
                            // smaller than 1 instead of 0, because insertion is counted as the previous position
                            pos_in_read += cg.len() as usize;
                            continue;
                        }
                        if col >= tensor_len as i64 {
                            break;
                        }
                        assert!(col >= 1 && col < tensor_len as i64, "Error: col out of range: {}", col);
                        tensor_row[(col - 1) as usize].insertion = 1;
                        pos_in_read += cg.len() as usize;
                    }
                    b'N' => {
                        for _ in 0..cg.len() {
                            if col < 0 {
                                col += 1;
                                continue;
                            }
                            if col >= tensor_len as i64 {
                                break;
                            }
                            assert!(col >= 0 && col < tensor_len as i64, "Error: col out of range: {}", col);
                            tensor_row[col as usize].base = 5;
                            tensor_row[col as usize].baseq = 0;
                            tensor_row[col as usize].mapq = mapq;
                            tensor_row[col as usize].strand = strand;
                            tensor_row[col as usize].ts_strand = ts_strand;
                            image_tensor.allele_cnts[col as usize][4] += 1;
                            col += 1;
                        }
                    }
                    _ => {
                        panic!("Error: unknown cigar operation: {}", cg.char());
                    }
                }
            }
            image_tensor.tensor.push(tensor_row);
        }
        image_tensor
    }


    pub fn append_reference(&mut self, references: &HashMap<String, Vec<u8>>) {
        /*
        Fill the ``ref_base`` field in each BaseFreq.
        Optional. If not called, the ``ref_base`` field in each BaseFreq will be '\0'
         */
        let chr = &self.region.chr;
        let s = self.region.start - 1;    // 0-based, inclusive
        let mut p = s as usize;
        for i in 0..self.freq_vec.len() {
            if self.freq_vec[i].i {
                self.freq_vec[i].ref_base = '-'; // insertion
            } else {
                self.freq_vec[i].ref_base = references.get(chr).unwrap()[p] as char;
                p += 1;
            }
        }
    }
}

pub fn independent_test(data: [u32; 4]) -> f64 {
    // use chi-square test or fisher's exact test to test whether two variables are independent
    // N<40 or T<1, use fisher's exact test
    // N>=40, T>=5 in >80% of cells, use chi-square test
    // N>=40, T>=5 in <=80% of cells, use fisher's exact test
    // return true if independent, false if not independent
    let mut phred_pvalue = 0.0;
    let N = data[0] + data[1] + data[2] + data[3];
    let mut TL = [0.0; 4];
    TL[0] = (((data[0] + data[1]) * (data[0] + data[2])) as f32) / (N as f32);
    TL[1] = (((data[0] + data[1]) * (data[1] + data[3])) as f32) / (N as f32);
    TL[2] = (((data[2] + data[3]) * (data[0] + data[2])) as f32) / (N as f32);
    TL[3] = (((data[2] + data[3]) * (data[1] + data[3])) as f32) / (N as f32);
    let mut T = 0.0;
    for i in 0..data.len() {
        if TL[i] >= 5.0 {
            T += 1.0;
        }
    }
    T = T / data.len() as f32;

    if N > 40 && T > 0.8 {
        // use chi-square test
        let x = vec![data[0] as f64, data[1] as f64];
        let y = vec![data[2] as f64, data[3] as f64];
        let p = ChiSquare::test_vector(&x, &y).p_value();
        phred_pvalue = -10.0 * p.log10();
    }

    if N <= 40 {
        // use fisher's exact test
        let p = fishers_exact(&data).unwrap().two_tail_pvalue;
        phred_pvalue = -10.0 * p.log10();
    }

    return phred_pvalue;
}