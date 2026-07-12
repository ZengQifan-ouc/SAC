phase_vcf=$1
bam_dir=$2
mask_dir=$3
train_dir=$4
output_dir=$5

#SNP calling
if [ ! -d "${bam_dir}/vcf_dir/" ]; then
    mkdir -p "${bam_dir}/vcf_dir/"
fi
gunzip ${train_dir}/${phase_vcf} > ${bam_dir}/vcf_dir/${phase_vcf%\.gz}
python SAC_phase.py --vcf ${bam_dir}/vcf_dir/${phase_vcf%\.gz} --bam-dir ${bam_dir} --genome genome/GRCh38_full_analysis_set_plus_decoy_hla.fa --output ${mask_dir}/${phase_vcf%_train.vcf.gz}_mask.csv
sed -i 's/\t/:/g' bam/vcf_dir/${phase_vcf%_train.vcf.gz}_train_header.txt 
grep -v "#" bam/vcf_dir/${phase_vcf%_train.vcf.gz}_train.vcf | cut -f 4,5 | paste bam/vcf_dir/${phase_vcf%_train.vcf.gz}_train_header.txt - | sed 's/\t/_/g' | sed ':a;N;s/\n/,/g;ta' | sed 's/^/bam_file,/g' > y.header
cd $bam_dir
ls *bam | sed 's/\.bam//g' | sed 's/$/_hap1/g' > x.header1
sed 's/hap1/hap2/g' x.header1x.header1 | paste x.header1 - | sed 's/\t/\n/g' > x.header
sed '1d' ${mask_dir}/${phase_vcf%_train.vcf.gz}_mask.csv | cut -f 2 -d \,  | paste x.header - | sed 's/\t/,/g' | cat y.header - > ${mask_dir}/${phase_vcf%_train.vcf.gz}_mask.new.csv
mv ${mask_dir}/${phase_vcf%_train.vcf.gz}_mask.new.csv ${mask_dir}/${phase_vcf%_train.vcf.gz}_mask.csv
rm x.header* y.header 
#reference panel format conversion
cd $train_dir
sh vcf2csv_train.sh
mkdir SAC_train
mv *csv SAC_train
cd ..
#imputation
python SAC_impute.py --train_csv ${train_dir}/SAC_train/${phase_vcf%\.vcf}.csv --shapeit_csv ${mask_dir}/${phase_vcf%_train.vcf.gz}_mask.csv --output_dir $output_dir