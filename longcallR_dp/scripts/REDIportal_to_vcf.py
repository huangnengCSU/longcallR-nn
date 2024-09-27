def write_REDIportal_vcf(REDIportal_file, output_vcf_file):
    fout = open(output_vcf_file,"a")
    fin = open(REDIportal_file, "r")
    # fout.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample\n")
    chroms = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']
    chroms = set(chroms)
    head = fin.readline()
    for line in fin:
        fields = line.strip().split('\t')
        chr = fields[0]
        pos = fields[1]
        id = "."
        ref = fields[2]
        alt = fields[3]
        qual = "10"
        flt = "PASS"
        info = "."
        fmt = "GT"
        sample = "0/1"
        if chr not in chroms:
            continue
        fout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(chr, pos, id, ref, alt, qual, flt, info, fmt, sample))
    fin.close()
    fout.close()