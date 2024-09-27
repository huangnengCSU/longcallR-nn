import os
from concurrent.futures import ProcessPoolExecutor

import pysam


# def filter_rna_edit_events(rna_edit_vcf, bam, pass_rna_edit_vcf, contig, flank_size, min_editing_ratio=0.1,
#                            min_edit_counts=2):
#     """
#     Filter RNA editing events based on the editing ratio
#     :param rna_edit_vcf: vcf file that contains RNA editing events
#     :param bam: long read RNA-seq bam file
#     :param pass_rna_edit_vcf: output vcf file that contains RNA editing events that pass the filtering
#     :param flank_size: the size of the flanking region for editing ratio calculation
#     :return:
#     """
#     # Load RNA editing event vcf file
#     vcf_file = pysam.VariantFile(rna_edit_vcf)
#     bam_file = pysam.AlignmentFile(bam, "rb")
#     # chromosome_names = list(vcf_file.header.contigs)
#     rna_edit_events_dict = {}
#     pass_edit_events_positions = []
#     # for chromosome in chromosome_names:
#     edit_positions = []
#     for record in vcf_file.fetch(contig):
#         chr = record.chrom
#         pos = record.pos - 1  # 0-based
#         rna_edit_events_dict[(chr + ":" + str(pos))] = record
#         edit_positions.append(pos)
#
#     for idx, pos in enumerate(edit_positions):
#         # pos_chunk, all positions within the flank_size for the current position edit_positions[idx]
#         pos_chunk = []
#
#         # left search
#         left_indices = []
#         i = idx - 1
#         while i >= 0 and pos - edit_positions[i] <= flank_size:
#             left_indices.append(i)
#             pos_chunk.append(edit_positions[i])
#             i -= 1
#
#         # right search
#         right_indices = []
#         j = idx + 1
#         while j < len(edit_positions) and edit_positions[j] - pos <= flank_size:
#             right_indices.append(j)
#             pos_chunk.append(edit_positions[j])
#             j += 1
#
#         # add the current position
#         pos_chunk.append(pos)
#
#         # sort pos_chunk
#         pos_chunk = sorted(pos_chunk)
#
#         left_most_pos = pos_chunk[0]
#         right_most_pos = pos_chunk[-1]
#
#         ## get the pileup reads for each position in pos_chunk
#         rna_edit_ratios = []
#         current_position_edit_ratio = 0.0
#         for pileupcolumn in bam_file.pileup(contig, left_most_pos, right_most_pos + 1):
#             if pileupcolumn.pos not in pos_chunk:
#                 continue
#             if pileupcolumn.nsegments == 0:
#                 continue
#             allele_counts = {"A": 0, "C": 0, "G": 0, "T": 0}
#             for pileupread in pileupcolumn.pileups:
#                 if pileupread.query_position is None:
#                     continue
#                 base = pileupread.alignment.query_sequence[pileupread.query_position].upper()
#                 if base not in ["A", "C", "G", "T"]:
#                     continue
#                 allele_counts[base] += 1
#             total_allele_count = sum(allele_counts.values())
#             if total_allele_count == 0 or (allele_counts['C'] == 0 and allele_counts['G'] == 0):
#                 continue
#             if allele_counts['C'] > allele_counts['G']:
#                 ## T -> C
#                 rna_edit_ratios.append(allele_counts['C'] / total_allele_count)
#                 if pileupcolumn.pos == pos:
#                     current_position_edit_ratio = allele_counts['C'] / total_allele_count
#             elif allele_counts['G'] > allele_counts['C']:
#                 ## A -> G
#                 rna_edit_ratios.append(allele_counts['G'] / total_allele_count)
#                 if pileupcolumn.pos == pos:
#                     current_position_edit_ratio = allele_counts['G'] / total_allele_count
#             else:
#                 print("uncertain editing event, position: {}:{}".format(contig, pileupcolumn.pos))
#
#         ## filter by editing ratio, at least two non-zero editing ratios
#         if current_position_edit_ratio >= min_editing_ratio and len(
#                 [ratio for ratio in rna_edit_ratios if ratio >= min_editing_ratio]) >= min_edit_counts:
#             print(chr + ":" + str(pos + 1))
#             pass_edit_events_positions.append(chr + ":" + str(pos))
#
#     with pysam.VariantFile(pass_rna_edit_vcf, "w", header=vcf_file.header) as pass_vcf:
#         for position in pass_edit_events_positions:
#             record = rna_edit_events_dict[position]
#             pass_vcf.write(record)


def load_pileup(bam_file, ctg, ref_seq):
    # create a list of lists to store the allele counts for each position
    allele_counts_list = [[0, 0, 0, 0] for _ in range(len(ref_seq))]
    bam = pysam.AlignmentFile(bam_file, "rb")
    print(f"Start loading pileup for {ctg}")

    for read in bam.fetch(ctg):
        ref_pos = read.reference_start  # 0-based
        read_seq = read.query_sequence
        read_pos = 0
        for op, length in read.cigartuples:
            if op == 0 or op == 7 or op == 8:
                for i in range(length):
                    base = read_seq[read_pos + i].upper()
                    if base not in ["A", "C", "G", "T"]:
                        continue
                    idx = ref_pos + i
                    if base == "A":
                        allele_counts_list[idx][0] += 1
                    elif base == "C":
                        allele_counts_list[idx][1] += 1
                    elif base == "G":
                        allele_counts_list[idx][2] += 1
                    elif base == "T":
                        allele_counts_list[idx][3] += 1
            if op in [0, 2]:
                ref_pos += length
            if op in [0, 1]:
                read_pos += length
            if op == 3:
                ref_pos += length
            if op == 4:
                read_pos += length

    print(f"Finished loading pileup for {ctg}")
    return allele_counts_list


def load_rna_edits(vcf_file, ctg):
    vcf = pysam.VariantFile(vcf_file)
    positions = []
    rna_edits = {}
    for record in vcf.fetch(ctg):
        chr = record.chrom
        pos = record.pos - 1  # 0-based
        rna_edits[(chr, pos)] = record
        positions.append(pos)
    return positions, rna_edits


def load_ref_seq(ref_file, ctg):
    fasta = pysam.FastaFile(ref_file)
    return fasta.fetch(ctg)


def filter_rna_edits(vcf_file, bam_file, ref_file, ctg, output_dir, flank_size, min_editing_ratio=0.1,
                     min_edit_counts=2):
    ref_seq = load_ref_seq(ref_file, ctg)
    allele_counts_list = load_pileup(bam_file, ctg, ref_seq)
    edit_positions, rna_edits = load_rna_edits(vcf_file, ctg)
    pass_edit_events_positions = []
    for idx, pos in enumerate(edit_positions):
        pos_chunk = []

        # left search
        left_indices = []
        i = idx - 1
        while i >= 0 and pos - edit_positions[i] <= flank_size:
            left_indices.append(i)
            pos_chunk.append(edit_positions[i])
            i -= 1

        # right search
        right_indices = []
        j = idx + 1
        while j < len(edit_positions) and edit_positions[j] - pos <= flank_size:
            right_indices.append(j)
            pos_chunk.append(edit_positions[j])
            j += 1

        # add the current position
        pos_chunk.append(pos)

        # sort pos_chunk
        pos_chunk = sorted(pos_chunk)

        rna_edit_ratios = []
        allele_cnts = allele_counts_list[pos]
        match ref_seq[pos]:
            case "A":
                if allele_cnts[2] == 0:
                    continue
            case "T":
                if allele_cnts[1] == 0:
                    continue
            case _:
                continue

        for p in pos_chunk:
            allele_cnts = allele_counts_list[p]
            if sum(allele_cnts) == 0:
                continue
            match ref_seq[p]:
                case "A":
                    ratio = allele_cnts[2] / sum(allele_cnts)
                    if ratio >= min_editing_ratio:
                        rna_edit_ratios.append(ratio)
                case "T":
                    ratio = allele_cnts[1] / sum(allele_cnts)
                    if ratio >= min_editing_ratio:
                        rna_edit_ratios.append(ratio)
                case _:
                    continue

        if len(rna_edit_ratios) >= min_edit_counts:
            pass_edit_events_positions.append((ctg, pos))

    if len(pass_edit_events_positions) > 0:
        header = pysam.VariantFile(vcf_file).header
        with pysam.VariantFile(output_dir + f"/{ctg}.vcf", "w", header=header) as fout:
            for ctg, pos in pass_edit_events_positions:
                record = rna_edits[(ctg, pos)]
                fout.write(record)


# def multiple_threads_run(vcf_file, bam_file, ref_file, output_dir, flank_size, min_editing_ratio=0.1,
#                          min_edit_counts=2, n_threads=1):
#     # get all references
#     fasta = pysam.FastaFile(ref_file)
#     references = fasta.references
#     with ThreadPoolExecutor(max_workers=n_threads) as executor:
#         for ctg in references:
#             executor.submit(filter_rna_edits, vcf_file, bam_file, ref_file, ctg, output_dir, flank_size,
#                             min_editing_ratio, min_edit_counts)


def multiple_threads_run(vcf_file, bam_file, ref_file, output_dir, flank_size, min_editing_ratio=0.1,
                         min_edit_counts=2, n_threads=1):
    # Get all references
    fasta = pysam.FastaFile(ref_file)
    references = fasta.references

    # Use ProcessPoolExecutor to utilize multiple CPU cores
    with ProcessPoolExecutor(max_workers=n_threads) as executor:
        for ctg in references:
            executor.submit(
                filter_rna_edits,
                vcf_file,
                bam_file,
                ref_file,
                ctg,
                output_dir,
                flank_size,
                min_editing_ratio,
                min_edit_counts
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter RNA editing events based on the editing ratio")
    parser.add_argument("-i", "--rna_edit_vcf", type=str, required=True,
                        help="Input vcf file that contains RNA editing events")
    parser.add_argument("-b", "--bam", type=str, required=True, help="Long read RNA-seq bam file")
    parser.add_argument("-r", "--ref", type=str, required=True, help="Reference genome fasta file")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory that contains the filtered RNA editing events")
    parser.add_argument("-f", "--flank_size", type=int, required=True,
                        help="The size of the flanking region for editing ratio calculation")
    parser.add_argument("--min_editing_ratio", type=float, default=0.1,
                        help="Minimum editing ratio for an editing event to be considered")
    parser.add_argument("--min_edit_counts", type=int, default=2,
                        help="Minimum editing counts for an editing event to be considered")
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    multiple_threads_run(args.rna_edit_vcf, args.bam, args.ref, args.output_dir, args.flank_size,
                         args.min_editing_ratio, args.min_edit_counts, args.threads)
