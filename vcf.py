def write_vcf_header(output_file, chromosome_lengths=None):
    header_lines = [
        "##fileformat=VCFv4.2",
        "##FILTER=<ID=PASS,Description=\"All filters passed\">",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">",
        "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">",
        "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">",
    ]

    if chromosome_lengths is None:
        # Dictionary of GRCh38 chromosomes and their lengths
        chromosome_lengths = {
            "chr1": 248956422,
            "chr2": 242193529,
            "chr3": 198295559,
            "chr4": 190214555,
            "chr5": 181538259,
            "chr6": 170805979,
            "chr7": 159345973,
            "chr8": 145138636,
            "chr9": 138394717,
            "chr10": 133797422,
            "chr11": 135086622,
            "chr12": 133275309,
            "chr13": 114364328,
            "chr14": 107043718,
            "chr15": 101991189,
            "chr16": 90338345,
            "chr17": 83257441,
            "chr18": 80373285,
            "chr19": 58617616,
            "chr20": 64444167,
            "chr21": 46709983,
            "chr22": 50818468,
            "chrX": 156040895,
            "chrY": 57227415,
            "chrMT": 16569
        }

    # Add a contig line for each chromosome with its length
    for chrom, length in chromosome_lengths.items():
        header_lines.append(f"##contig=<ID={chrom},length={length}>")

    # Add the column header line
    header_lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1")

    with open(output_file, 'w') as vcf_file:
        for line in header_lines:
            vcf_file.write(line + "\n")
