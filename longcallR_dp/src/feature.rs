use mathru::algebra::abstr::cast::ToPrimitive;

#[derive(Debug, Clone, Default)]
pub struct Feature {
    pub base_cnt: [u32; 7],   // A, C, G, T, N, Del, Ins
    pub ref_base: u8,         // 0, 1, 2, 3, 4, 5
    pub avg_baseq: [f32; 4],    // A, C, G, T,
    pub avg_mapq: [f32; 4],   // A, C, G, T,
    pub forward_cnt: [u32; 4],  // A, C, G, T
    pub reverse_cnt: [u32; 4],  // A, C, G, T
    pub ts_forward_cnt: u32,  // forward transcript strand
    pub ts_reverse_cnt: u32,  // reverse transcript strand
}

impl Feature {
    pub fn to_csv(&self) -> String {
        // write all fields to csv as a line
        let mut csv_line = String::new();
        for item in self.base_cnt.iter() {
            csv_line.push_str(&item.to_f32().to_string());
            csv_line.push(',');
        }
        csv_line.push_str(&self.ref_base.to_f32().to_string());
        csv_line.push(',');
        for item in self.avg_baseq.iter() {
            csv_line.push_str(&item.to_string());
            csv_line.push(',');
        }
        for item in self.avg_mapq.iter() {
            csv_line.push_str(&item.to_string());
            csv_line.push(',');
        }
        for item in self.forward_cnt.iter() {
            csv_line.push_str(&item.to_f32().to_string());
            csv_line.push(',');
        }
        for item in self.reverse_cnt.iter() {
            csv_line.push_str(&item.to_f32().to_string());
            csv_line.push(',');
        }
        csv_line.push_str(&self.ts_forward_cnt.to_f32().to_string());
        csv_line.push(',');
        csv_line.push_str(&self.ts_reverse_cnt.to_f32().to_string());
        csv_line.push(',');
        return csv_line;
    }
}