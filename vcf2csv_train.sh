for i in `ls *.vcf.gz`; do plink --vcf $i --recode 01 --allow-extra-chr --out ${i%\.vcf.gz}  --output-missing-genotype 2; done
for i in `ls *map`; do cut -f 1,4 $i | sed 's/\t/_/g' | sed 's/$/_1/g' > ${i}1; cut -f 1,4 $i | sed 's/\t/_/g' | sed 's/$/_2/g' > ${i}2; done
for i in `ls *map`; do paste ${i}1 ${i}2 | sed 's/\t/,/g' | sed ':a;N;s/\n/,/g;ta' > ${i%map}header; done
test_vcf=`ls *vcf.gz | head -1`
samp_num=`zcat $test_vcf | grep "#CHROM" | cut -f 11- | wc -w`
for i in $(seq 0 $samp_num); do echo $i >> y.header; done
sed -i '1s/^/\n/' y.header
for i in `ls *.vcf.gz`; do zcat $i | grep -v "#" | cut -f 10- | awk '{for (i=1; i<=NF; i++) {a[NR,i] = $i}}END {for (j=1; j<=NF; j++) {for (i=1; i<=NR; i++) {printf "%s%s", a[i,j], (i==NR ? "\n" : " ")}}}' | sed 's/|/ /g' | sed 's/ /,/g' | cat ${i%\.vcf.gz}.header - | paste y.header - | sed 's/\t/,/g' > ${i%\.vcf.gz}.csv; done

