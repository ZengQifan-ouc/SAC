Genotype Phasing & Imputation Pipeline
Two Python scripts for haplotype phasing and genotype imputation using deep learning.

Files:
phase_script.py - Phases haplotypes using BAM files and a reference panel
impute_script.py - Imputes missing genotypes using a CNN + Transformer model

Dependencies:
numpy pandas torch pysam cyvcf2 numba tqdm bgzip

Input Files:
VCF file (reference haplotype panel)
BAM files (target samples, with .bai index)
Reference genome FASTA (with .fai index)

Usage:
Step 1: Phasing
python phase_script.py --vcf reference.vcf --bam-dir /bam/folder --genome genome.fa --output phased.csv
Step 2: Imputation
python impute_script.py --train_csv phased.csv --test_csv test_data.csv --shapeit_csv shapeit.csv --output_dir ./results

Output:
phased.csv - Phased haplotypes in CSV format
*.imputed_SAC.vcf.gz - Imputed genotypes in compressed VCF format with confidence scores

Notes:
GPU is optional but recommended for imputation
Use 32 cores for phasing (adjustable in the code)
BAM files must be indexed (samtools index sample.bam)
