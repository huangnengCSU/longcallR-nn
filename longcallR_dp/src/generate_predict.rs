use std::collections::{HashSet, VecDeque};
use std::fs;
use std::fs::File;
use std::hash::Hash;
use std::sync::Mutex;

use ndarray_npy::NpzWriter;
use rayon::prelude::*;
use rust_htslib::bam::{IndexedReader, Read};

use crate::util::{find_covering_region_indices, load_reference, parse_fai, Profile, Region};

pub fn generate(bam_file: String, ref_file: String, output_dir: String, isolated_regions: Vec<Region>, min_mapq: u8,
                min_baseq: u8, min_read_length: usize, min_alt_freq: f32, min_alt_depth: u32, min_depth: u32, max_depth: u32, flanking_size: u32, chunk_size: u32, thread_size: usize) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(thread_size).build().unwrap();
    let feature_queue = Mutex::new(VecDeque::new());
    let ref_seqs = load_reference(ref_file.clone());
    let fai_path = ref_file + ".fai";
    if fs::metadata(&fai_path).is_err() {
        panic!("Reference index file .fai does not exist.");
    }
    let contig_lengths = parse_fai(fai_path.as_str());
    let mut contig_order = Vec::new();
    for (k, _) in contig_lengths.iter() {
        contig_order.push(k.clone());
    }

    pool.install(|| {
        isolated_regions.par_iter().for_each(|reg| {
            let mut profile = Profile::default();
            let ref_seq = ref_seqs.get(&reg.chr).unwrap();

            // load all reads in the region
            let mut bam = IndexedReader::from_path(&bam_file).expect("Error opening BAM file");
            let tid = bam.header().tid(reg.chr.as_bytes()).unwrap();
            let start: i64 = reg.start as i64 - 1;   // 0-based, inclusive
            let end: i64 = reg.end as i64 - 1;   // 0-based, exclusive
            bam.fetch((tid, start, end)).expect("Error fetching reads");    // fetch is zero-based. start is inclusive, stop is exclusive.

            let mut reads_vec = Vec::new();
            for result in bam.records() {
                let record = result.expect("Error reading record");
                if record.is_unmapped() || record.is_secondary() || record.is_supplementary() || record.mapq() < min_mapq {
                    continue;
                }
                reads_vec.push(record);
            }

            // image range with 2*flanking_size
            /*
            flank_size = 3
            ACAGTCGTCGGAC
                   ||||||
                   TCGGACTCTTC
            */

            let mut chunk_regions = Vec::new();
            let mut chunk_regions_intervals = Vec::new();   // 1-based, left inclusive, right exclusive
            let mut current_start = reg.start;
            while current_start < reg.end {
                // Calculate the end of the current chunk
                let current_end = if current_start + chunk_size >= reg.end {
                    reg.end
                } else {
                    current_start + chunk_size
                };
                let chunk_region = Region {
                    chr: reg.chr.clone(),
                    start: current_start,
                    end: current_end,
                    gene_id: None,
                };
                chunk_regions.push(chunk_region);
                chunk_regions_intervals.push((current_start as usize, current_end as usize));
                if current_end >= reg.end {
                    break;
                }
                // Move to the next chunk with the overlap
                current_start += chunk_size - 2 * flanking_size;
                // If the next chunk start is less than the end, adjust it to not exceed the end
                if current_start + chunk_size >= reg.end && current_start < reg.end {
                    let chunk_region = Region {
                        chr: reg.chr.clone(),
                        start: current_start,
                        end: reg.end,
                        gene_id: None,
                    };
                    chunk_regions.push(chunk_region);
                    chunk_regions_intervals.push((current_start as usize, reg.end as usize));
                    break;
                }
            }

            // assign read to each chunk, some reads may be assigned to multiple chunks
            let mut chunk_read_idxes: Vec<HashSet<usize>> = vec![HashSet::new(); chunk_regions_intervals.len()];
            for (i, read) in reads_vec.iter().enumerate() {
                let pos = read.pos() as usize;
                let cigar = read.cigar();
                let leading_softclips = cigar.leading_softclips();
                let mut pos_in_ref = pos;
                let mut pos_in_read = if leading_softclips > 0 { leading_softclips as usize } else { 0 };
                for &cigar_elem in cigar.iter() {
                    let len = cigar_elem.len() as usize;
                    match cigar_elem.char() {
                        'M' | '=' | 'X' => {
                            // for _ in 0..len {
                            //     let hit_idxes = find_covering_region_indices(&chunk_regions_intervals, pos_in_ref);
                            //     for hit_idx in hit_idxes {
                            //         chunk_read_idxes[hit_idx].insert(i);
                            //     }
                            //     pos_in_ref += 1;
                            //     pos_in_read += 1;
                            // }
                            let left_pos = pos_in_ref;  // 0-based, inclusive
                            pos_in_ref += len;
                            pos_in_read += len;
                            let right_pos = pos_in_ref; // 0-based, exclusive
                            if (left_pos as i64) >= end || (right_pos as i64) <= start {
                                continue;
                            }
                            let left_pos = if left_pos < start as usize { start as usize } else { left_pos };   // 0-based, inclusive
                            let right_pos = if right_pos >= end as usize { end as usize } else { right_pos };    // 0-based, exclusive
                            let hit_idxes1 = find_covering_region_indices(&chunk_regions_intervals, left_pos + 1);
                            let hit_idxes2 = find_covering_region_indices(&chunk_regions_intervals, right_pos);
                            assert!(hit_idxes1.len() > 0 && hit_idxes2.len() > 0, "readname: {:?}\nregion: {}:{}-{}\nleft_pos: {}, right_pos: {}\nchunk_regions:\n{:?}", std::str::from_utf8(read.qname()), reg.chr, reg.start, reg.end, left_pos, right_pos, &chunk_regions_intervals);
                            for hit_idx in hit_idxes1[0]..=hit_idxes2[hit_idxes2.len() - 1] {
                                chunk_read_idxes[hit_idx].insert(i);
                            }
                        }
                        'I' => {
                            pos_in_read += len;
                        }
                        'D' | 'N' => {
                            pos_in_ref += len;
                        }
                        _ => {}
                    }
                }
            }

            for chunk_idx in 0..chunk_regions.len() {
                let chunk_region = &chunk_regions[chunk_idx];
                let read_idxes = &chunk_read_idxes[chunk_idx];
                // calculate the bam to image runtime for each chunk
                // println!("Processing chunk {}:{}-{}", chunk_region.chr, chunk_region.start, chunk_region.end);
                // let start_time = std::time::Instant::now();
                if read_idxes.len() == 0 {
                    // no reads are assigned to the chunk
                    continue;
                }
                // let image_tensor = profile.bam_to_image(&bam_file, &chunk_region, ref_seq, min_mapq);
                let image_tensor = profile.bam_to_image2(&reads_vec, read_idxes, &chunk_region, ref_seq, min_baseq, min_mapq, max_depth);
                // let elapsed = start_time.elapsed();
                // if image_tensor.tensor.len() == 0 {
                //     println!("Elapsed time: {} seconds, shape: 0,0", elapsed.as_secs_f32());
                // } else {
                //     println!("Elapsed time: {} seconds, shape: {},{}", elapsed.as_secs_f32(), image_tensor.tensor.len(), image_tensor.tensor[0].len());
                // }
                let image_start = chunk_region.start;   // 1-based, inclusive
                let image_end = chunk_region.end;   // 1-based, exclusive
                if chunk_idx == 0 {
                    // first chunk, the front positions may not have enough flanking size, will pad the left side
                    for locus in image_start..image_end - flanking_size {
                        // let (alt_fraction, alt_cnt, depth) = image_tensor.get_alt_fraction_depth((locus - image_start) as usize, min_baseq);
                        let (alt_fraction, alt_cnt, depth) = image_tensor.get_alt_fraction_depth2((locus - image_start) as usize);
                        if depth as u32 >= min_depth && alt_fraction >= min_alt_freq && alt_cnt as u32 >= min_alt_depth {
                            let matrix = image_tensor.get_feature_maxrix((locus - image_start) as usize, flanking_size);
                            // let matrix = image_tensor.get_feature_maxrix2((locus - image_start) as usize, flanking_size, max_depth);
                            feature_queue.lock().unwrap().push_back((matrix, format!("{}:{}", &reg.chr, (&locus).to_string())));
                        }
                    }
                } else if chunk_idx == chunk_regions.len() - 1 {
                    // last chunk, the end positions may not have enough flanking size, will pad the right side
                    for locus in image_start + flanking_size..image_end {
                        // let (alt_fraction, alt_cnt, depth) = image_tensor.get_alt_fraction_depth((locus - image_start) as usize, min_baseq);
                        let (alt_fraction, alt_cnt, depth) = image_tensor.get_alt_fraction_depth2((locus - image_start) as usize);
                        if depth as u32 >= min_depth && alt_fraction >= min_alt_freq && alt_cnt as u32 >= min_alt_depth {
                            let matrix = image_tensor.get_feature_maxrix((locus - image_start) as usize, flanking_size);
                            // let matrix = image_tensor.get_feature_maxrix2((locus - image_start) as usize, flanking_size, max_depth);
                            feature_queue.lock().unwrap().push_back((matrix, format!("{}:{}", &reg.chr, (&locus).to_string())));
                        }
                    }
                } else {
                    // middle chunk, no padding needed
                    for locus in image_start + flanking_size..image_end - flanking_size {
                        // let (alt_fraction, alt_cnt, depth) = image_tensor.get_alt_fraction_depth((locus - image_start) as usize, min_baseq);
                        let (alt_fraction, alt_cnt, depth) = image_tensor.get_alt_fraction_depth2((locus - image_start) as usize);
                        if depth as u32 >= min_depth && alt_fraction >= min_alt_freq && alt_cnt as u32 >= min_alt_depth {
                            let matrix = image_tensor.get_feature_maxrix((locus - image_start) as usize, flanking_size);
                            // let matrix = image_tensor.get_feature_maxrix2((locus - image_start) as usize, flanking_size, max_depth);
                            feature_queue.lock().unwrap().push_back((matrix, format!("{}:{}", &reg.chr, (&locus).to_string())));
                        }
                    }
                }
            }
        });
    });
    let array_file = File::create(output_dir.clone() + "/predict.npz").unwrap();
    let mut npz = NpzWriter::new(array_file);
    while !feature_queue.lock().unwrap().is_empty() {
        let feature_vec = feature_queue.lock().unwrap().pop_front().unwrap();
        npz.add_array(&feature_vec.1, &feature_vec.0);
    }
    npz.finish().unwrap();
}