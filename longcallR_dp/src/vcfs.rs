use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use lazy_static::lazy_static;
use noodles_vcf;
use noodles_vcf::variant::record::{AlternateBases, Filters};
use noodles_vcf::variant::record::samples::keys::key;
use noodles_vcf::variant::record::samples::Sample;
use noodles_vcf::variant::record::samples::series::Value;

lazy_static! {
    pub static ref GT_MAP: HashMap<&'static str, u8> = {
        let mut m = HashMap::new();
        m.insert("AA", 0);
        m.insert("AC", 1);
        m.insert("AG", 2);
        m.insert("AT", 3);
        m.insert("CA", 4);
        m.insert("CC", 5);
        m.insert("CG", 6);
        m.insert("CT", 7);
        m.insert("GA", 8);
        m.insert("GC", 9);
        m.insert("GG", 10);
        m.insert("GT", 11);
        m.insert("TA", 12);
        m.insert("TC", 13);
        m.insert("TG", 14);
        m.insert("TT", 15);
        m
    };
}

pub fn load_vcf(vcf_path: &String) -> (HashMap<String, HashSet<usize>>, HashMap<String, HashMap<usize, (u8, u8)>>) {
    let mut input_candidates: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut input_candidates_genotype: HashMap<String, HashMap<usize, (u8, u8)>> = HashMap::new();
    let mut reader = noodles_vcf::io::reader::Builder::default().build_from_path(vcf_path).unwrap();
    let header = reader.read_header().unwrap();
    for result in reader.records() {
        let record = result.unwrap();
        let ref_base = record.reference_bases().to_string();
        let alt_bases = record.alternate_bases().iter().map(|x| x.unwrap().to_string()).collect::<Vec<String>>();
        if ref_base.len() != 1 || alt_bases.iter().any(|x| x.len() != 1) {
            continue;
        }
        if record.filters().is_pass(&header).unwrap() {
            let chr = record.reference_sequence_name();
            let pos = record.variant_start().unwrap().unwrap().get();
            // store position by chromosome
            let samples = record.samples();
            for (_, sample) in samples.iter().enumerate() {
                let gt = sample.get(&header, key::GENOTYPE).unwrap().unwrap().unwrap();
                match gt {
                    Value::Genotype(genotype) => {
                        let mut gt_vec = Vec::new();
                        for vi in genotype.iter() {
                            let (allele, _) = vi.unwrap();
                            let allele = allele.unwrap();
                            gt_vec.push(allele);
                        }
                        if gt_vec.len() != 2 {
                            continue;
                        }
                        let r = ref_base.as_str();
                        let a = alt_bases[0].as_str();
                        let r_a = format!("{}{}", r, a);
                        if !GT_MAP.contains_key(r_a.as_str()) {
                            println!("{} not in GT_MAP", r_a.as_str());
                            continue;
                        }
                        let gt = GT_MAP.get(r_a.as_str()).unwrap();
                        if gt_vec[0] + gt_vec[1] == 1 {
                            // gt: 0/1
                            input_candidates_genotype.entry(chr.to_string()).or_insert(HashMap::new()).insert(pos, (1, *gt));
                            input_candidates.entry(chr.to_string()).or_insert(HashSet::new()).insert(pos);
                        } else if gt_vec[0] + gt_vec[1] == 2 {
                            // gt: 1/1
                            input_candidates_genotype.entry(chr.to_string()).or_insert(HashMap::new()).insert(pos, (2, *gt));
                            input_candidates.entry(chr.to_string()).or_insert(HashSet::new()).insert(pos);
                        } else if gt_vec[0] + gt_vec[1] == 3 {
                            // gt: 1/2
                            input_candidates_genotype.entry(chr.to_string()).or_insert(HashMap::new()).insert(pos, (3, *gt));
                            input_candidates.entry(chr.to_string()).or_insert(HashSet::new()).insert(pos);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    return (input_candidates, input_candidates_genotype);
}