import pysam
import pandas as pd
import os
from glob import glob


def process_vcf(vcf_path,input_dir):
    """处理单个VCF文件的函数"""
    try:
        # 打开VCF文件
        vcf = pysam.VariantFile(vcf_path)

        # 获取样本列表和染色体名称
        samples = list(vcf.header.samples)
        # 从第一个记录获取染色体名称
        first_record = next(iter(vcf))
        chrom = first_record.chrom
        # 重置迭代器
        vcf = pysam.VariantFile(vcf_path)

        haps_data = {sample: {'hap1': [], 'hap2': []} for sample in samples}
        cols = []

        # 遍历每个位点并处理基因型
        for record in vcf:
            # 获取REF和ALT信息
            ref = record.ref
            alts = record.alts
            alt1 = alts[0] if alts else '.'  # 如果没有ALT，用'.'表示
            
            # 构造位点列名，包含染色体、位置和REF_ALT信息
            pos_str = "{}:{}_{}_{}".format(chrom, record.pos, ref, alt1)
            cols.append(pos_str)

            for sample in samples:
                # 获取基因型
                gt = record.samples[sample].get('GT', None)

                # 处理缺失或杂合基因型
                if gt is None or None in gt:
                    h1, h2 = -1, -1
                else:
                    h1, h2 = gt

                # 存储单倍型数据
                haps_data[sample]['hap1'].append(h1)
                haps_data[sample]['hap2'].append(h2)

        # 构建数据框的行和行名
        rows = []
        row_names = []
        for sample in samples:
            rows.append(haps_data[sample]['hap1'])
            rows.append(haps_data[sample]['hap2'])
            row_names.append("{}_hap1".format(sample))
            row_names.append("{}_hap2".format(sample))

        # 创建输出文件名（使用兼容性写法）
        base_name = os.path.basename(vcf_path).replace('.vcf.gz', '')
        output_csv = os.path.join(input_dir, "{}.csv".format(base_name))

        # 创建DataFrame并保存为CSV
        df = pd.DataFrame(rows, index=row_names, columns=cols)
        df.to_csv(output_csv)

        print("成功处理: {} -> {}".format(vcf_path, output_csv))

    except Exception as e:
        print("处理 {} 时出错: {}".format(vcf_path, str(e)))
    finally:
        if 'vcf' in locals():
            vcf.close()








if __name__ == "__main__":
    # 设置输入目录
    input_dir = "/data/Linux/user/qc/data/human/mask_0.9/"
    # 查找所有.vcf.gz文件
    vcf_files = glob(os.path.join(input_dir, "*.vcf.gz"))

    if not vcf_files:
        print("在 {} 目录下未找到.vcf.gz文件".format(input_dir))
    else:
        print("找到 {} 个VCF文件待处理:".format(len(vcf_files)))
        for vcf_file in vcf_files:
            print("  - {}".format(vcf_file))

        # 处理每个VCF文件
        for vcf_file in vcf_files:
            process_vcf(vcf_file,input_dir)

        print("所有文件处理完成")


