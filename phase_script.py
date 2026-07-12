import os
import pysam 
import pandas as pd
import numba
from numba import njit, prange
import numpy as np
import time
from collections import Counter,defaultdict
from pathlib import Path
from tqdm import tqdm
import ctypes
import argparse
from cyvcf2 import VCF
from multiprocessing import Pool
import functools


def fast_vcf_to_txt(vcf_file, out_txt):
    vcf = VCF(vcf_file)
    header_file = out_txt.replace(".txt", "_header.txt")

    translate_allele = lambda a: '.' if a == -1 else str(a)

    with open(out_txt, "w") as f_out, open(header_file, "w") as f_header:
        for variant in tqdm(vcf, desc="🚀 高速解析 VCF"):
            if "GT" not in (variant.FORMAT or ""):
                continue

            # 使用整数型 genotypes（最快）
            gts = variant.genotypes  # List[List[int, int, bool]]
            row = ''.join(
                f"{translate_allele(gt[0])}{translate_allele(gt[1])}"
                for gt in gts
            )
            f_out.write(row + "\n")
            f_header.write(f"{variant.CHROM}\t{variant.POS}\n")

    print(f"\n✅ 完成：{out_txt}")

def fast_vcf_to_txt_full(vcf_file, out_txt):
    vcf = VCF(vcf_file)
    header_file = out_txt.replace(".txt", "_header.txt")

    translate_allele = lambda a: '.' if a == -1 else str(a)

    with open(out_txt, "w") as f_out, open(header_file, "w") as f_header:
        for variant in tqdm(vcf, desc="🚀 高速解析 VCF"):
            if "GT" not in (variant.FORMAT or ""):
                continue

            # 使用整数型 genotypes（最快）
            gts = variant.genotypes  # List[List[int, int, bool]]
            row = ''.join(
                f"{translate_allele(gt[0])}{translate_allele(gt[1])}"
                for gt in gts
            )
            f_out.write(row + "\n")
            f_header.write(f"{variant.CHROM}\t{variant.POS}\t{variant.REF}\t{variant.ALT}\n")

    print(f"\n✅ 完成：{out_txt}")


def build_line_index(file_path, index_path="offsets.idx"):
    """构建每行起始字节的偏移索引"""
    offsets = []
    offset = 0
    with open(file_path, 'rb') as f:
        for line in f:
            offsets.append(offset)
            offset += len(line)
    # 保存索引
    #with open(index_path, 'w') as f:
       # f.write('\n'.join(map(str, offsets)))
    return offsets

def load_offsets(index_path):
    with open(index_path, 'r') as f:
        return [int(line.strip()) for line in f]

def read_genotype_by_offset(file_path, offsets, start_line, num_lines=100):
    result = []
    with open(file_path, 'rb') as f:
        f.seek(offsets[start_line])
        for i in range(num_lines):
            line = f.readline()
            result.append(line[:-1])
        result= np.array([np.frombuffer(b, dtype=np.uint8) - ord('0') for b in result]).transpose()
    return result   


def read_header_by_offset(file_path, offsets, start_line, num_lines=100):
    result = []
    with open(file_path, 'r') as f:
        f.seek(offsets[start_line])
        for i in range(num_lines):
            line = f.readline()[:-1].replace('\t', '_')
            result.append(line)
    return result 

def get_temp_panel(file_path_genotype, offsets_genotype,file_path_header ,offset_header ,start_line, num_lines=100):

    ##########
    result = read_genotype_by_offset(file_path_genotype, offsets_genotype, start_line, num_lines)
    header = read_header_by_offset(file_path_header, offset_header, start_line, num_lines)
    #####################
    temp_panel= pd.DataFrame(result, columns=header)
    #first_parts = header[0].split('_')
    #chorm = '_'.join(first_parts[:2])  # 合并前两部分："NC_027300.1"
    #start = int(first_parts[2])        # 第一个位置
    #last_parts = header[-1].split('_')
    #end = int(last_parts[2])
    #############
    chorm=header[0].split('_')[0]
    start=int(header[0].split('_')[1])
    end= int(header[-1].split('_')[1])    
    return temp_panel,(chorm,start,end)


#获取区域内所有snp
def get_gold_snp(bamfile,chrom,start,end):
    # 获取区域内的reads
    golbal_snp={}
    for read in bamfile.fetch(chrom, start, end):
        if read.is_unmapped:
            continue  # 跳过未比对的读取
        # read_name = read.query_name
        # 初始化读取的 SNP 列表
        snp_list = []
        # 获取读取与参考的比对位置对
        aligned_pairs = read.get_aligned_pairs(matches_only=False, with_seq=True)
        start_pos = read.reference_start + 1
        end_pos = read.reference_end
        snp_list.append({"chrom":chrom,"start_pos": start_pos, "end_pos": end_pos})
        # snp_list.append([])
        # temp_snp=[[],[],[]]
        for query_pos, ref_pos, ref_base in aligned_pairs:
            # 跳过插入和缺失位置
            if query_pos is None or ref_pos is None or ref_base is None:
                continue
            # 检查位点是否在指定区域内
            if ref_pos+1 < start or ref_pos+1 >= end:
                continue  # 跳过不在区域内的位点
            # 获取读取和参考的碱基
            read_base = read.query_sequence[query_pos]
            ref_base = ref_base.upper()
            # 比较碱基，识别 SNP
            if read_base != ref_base:
                if chrom+"_"+str(ref_pos+1) not in golbal_snp:
                    golbal_snp[chrom+"_"+str(ref_pos+1)]=ref_base
    return golbal_snp


#去除golbal_snp中非snp的位点【一般认为snp独立存在，去掉连续的位点】
def remove_consecutive_values(golbal_snp,chrom):
    #golbal_snp 为函数 get_gold_snp 的输出 字典
    # chrom 为染色体号
    # return 去除连续的位点的字典 也就是 经过筛选的golbal_snp
    golbal_snp_=list(golbal_snp.keys())
    lst=[int(i.replace(chrom+"_","")) for i in golbal_snp_]
    lst = sorted(set(lst))
    result = {}
    i = 0
    while i < len(lst):
        if i == 0 or lst[i] - lst[i-1] != 1:
            # 检查下一个数字是否也是连续的
            if i + 1 < len(lst) and lst[i + 1] - lst[i] == 1:
                i += 1
                while i + 1 < len(lst) and lst[i + 1] - lst[i] == 1:
                    i += 1
            else:
                result[chrom+"_"+str(lst[i])]=golbal_snp[chrom+"_"+str(lst[i])]
        i += 1
    return result            


# 获取每条reads 的SNP连锁信息和对应的测序质量
def get_reads_snp_linkage(panel, bamfile, golbal_snp, chrom, start, end):
    """
    提取读取上的 SNP 连锁信息。
    返回 dict: read_name -> [meta_info_dict, [pos[], ref[], read[], qual[]]]
    """
    # 初始化 SNP 位点标记（可选操作，但逻辑看起来多余？）
    panel_snp_keys = set(panel.columns)
    for key in panel_snp_keys:
        golbal_snp[key] = "P"

    reads_snp_linkage = {}

    fetch_iter = bamfile.fetch(chrom, start, end)
    for read in fetch_iter:
        if read.is_unmapped:
            continue

        read_name = read.query_name
        aligned_pairs = read.get_aligned_pairs(matches_only=False, with_seq=True)

        temp_snp = [[], [], [], []]
        for query_pos, ref_pos, ref_base in aligned_pairs:
            if query_pos is None or ref_pos is None or ref_base is None:
                continue

            pos_1b = ref_pos + 1
            if pos_1b < start or pos_1b > end:
                continue

            key = f"{chrom}_{pos_1b}"
            if key in golbal_snp:
                read_base = read.query_sequence[query_pos]
                quality = read.query_qualities[query_pos]

                temp_snp[0].append(pos_1b)
                temp_snp[1].append(ref_base.upper())
                temp_snp[2].append(read_base)
                temp_snp[3].append(quality)

        if temp_snp[0]:  # 如果有 SNP 被收集
            meta = {"chrom": chrom, "start_pos": read.reference_start + 1, "end_pos": read.reference_end}
            reads_snp_linkage[read_name] = [meta, temp_snp]

    return reads_snp_linkage

# 获取reads的SNP矩阵和对应的测序质量

def get_snp_mat_quality(reads_snp_linkage, golbal_snp, chrom):
    """
    获取 SNP矩阵和测序质量矩阵（优化版本）
    """
    reads_names = list(reads_snp_linkage.keys())
    snp_positions = list(golbal_snp.keys())
    num_reads = len(reads_names)
    num_snps = len(snp_positions)

    # 初始化矩阵
    snp_mat = np.zeros((num_reads, num_snps), dtype=int)
    snp_mat_quality = np.zeros((num_reads, num_snps), dtype=int)

    # 构建 SNP 位置到索引的映射
    snp_pos_to_idx = {pos: idx for idx, pos in enumerate(snp_positions)}

    # 构建 read 名称到索引的映射
    read_name_to_idx = {name: idx for idx, name in enumerate(reads_names)}

    # 遍历每个 read 并填充矩阵
    for read_name, read_inf in reads_snp_linkage.items():
        read_idx = read_name_to_idx[read_name]

        if len(read_inf) in (2, 4):
            indices = [1] if len(read_inf) == 2 else [1, 3]

            for idx in indices:
                positions = read_inf[idx][0]
                ref_bases = read_inf[idx][1]
                read_bases = read_inf[idx][2]
                qualities = read_inf[idx][3]

                for i in range(len(positions)):
                    snp_pos = f"{chrom}_{positions[i]}"
                    if snp_pos in snp_pos_to_idx:
                        snp_idx = snp_pos_to_idx[snp_pos]
                        snp_mat[read_idx, snp_idx] = (ref_bases[i] != read_bases[i]) + 1
                        snp_mat_quality[read_idx, snp_idx] = qualities[i]

    # 转换为 DataFrame
    snp_mat_df = pd.DataFrame(snp_mat, index=reads_names, columns=snp_positions)
    snp_mat_quality_df = pd.DataFrame(snp_mat_quality, index=reads_names, columns=snp_positions)

    return snp_mat_df, snp_mat_quality_df

def get_panel_haptype_snp_mat_quality(snp_mat,snp_mat_quality,panel):
    panel_all=panel
    panle_snp_list=  panel_all.columns.tolist()
    panle_snp_list_index=[]
    all_snp_name= list(snp_mat.columns)
    
    for i in panle_snp_list:
        panle_snp_list_index.append( all_snp_name.index(i))

    snp_mat_panle= snp_mat.loc[:,panle_snp_list]
    

    usefull_read= (snp_mat_panle.sum(axis=1)!=0).to_list()

    # 先判断 snp_mat_panle 是否全为0
    if (snp_mat_panle == 0).all().all():
        # 若全为0，则不筛选，保留所有行
        usefull_read = [True] * len(snp_mat_panle)  # 全为True的列表，保留所有行
    else:
        # 若不全为0，则按原逻辑筛选有效行（排除全0行）
        usefull_read = (snp_mat_panle.sum(axis=1) != 0).to_list()

    # 应用筛选条件（无论是否全为0，都能安全执行）
    snp_mat = snp_mat.iloc[usefull_read, :]
    snp_mat_quality = snp_mat_quality.iloc[usefull_read, :]
    snp_mat_panle = snp_mat_panle.iloc[usefull_read, :]
    snp_mat_panle_quality = snp_mat_quality.loc[:, panle_snp_list]

    return snp_mat,snp_mat_quality,snp_mat_panle,panel_all,panle_snp_list_index,panle_snp_list,snp_mat_panle_quality

def get_overlap_group(snp_mat_):
    snp_mat=snp_mat_.copy()
    overlap_group=[]
    for i in range(snp_mat.shape[0]):
        pass
    i=0
    while i<=snp_mat.shape[0]-1:
        temp_group=[i]
        temp_i= snp_mat.iloc[i,:]
        flag=True
        while flag & (i<snp_mat.shape[0]-1):
            j=i+1
            temp_j= snp_mat.iloc[j,:]
            boolij= (temp_i>0) & (temp_j>0)
            if boolij.sum()>0:
                temp_group.append(j)
                i=j
                temp_i += temp_j
                flag=True
            else:
                i+=1
                flag=False
        overlap_group.append(temp_group)
        if i==snp_mat.shape[0]-1:
            i+=1
    return overlap_group


@njit(parallel=True)
def _compute_read_panel_similarity(snp_mat, quality_p, panel_all_np):
    n_reads, n_snps = snp_mat.shape
    n_haplotypes = panel_all_np.shape[0]

    result = np.zeros((n_reads, n_haplotypes), dtype=np.float64)

    for i in prange(n_reads):
        for j in range(n_haplotypes):
            log_score = 0.0
            for k in range(n_snps):
                snp_val = snp_mat[i, k]
                if snp_val < 0:
                    continue  # 缺失位点，不计入
                panel_val = panel_all_np[j, k]
                q = quality_p[i, k]
                if snp_val == panel_val:
                    log_score += np.log(1.0 - q)
                else:
                    log_score += np.log(q)
            result[i, j] = np.exp(log_score)
    return result

def get_reads_p_for_every_k(snp_mat_panle, snp_mat_panle_quality_p, panel_all):
    """
    获取每条reads与panel中每个单倍型的相似度（优化版本，带 numba 加速）
    """
    if not isinstance(snp_mat_panle, pd.DataFrame) or not isinstance(snp_mat_panle_quality_p, pd.DataFrame) or not isinstance(panel_all, pd.DataFrame):
        raise ValueError("All input parameters must be pandas DataFrames.")
    if snp_mat_panle.shape != snp_mat_panle_quality_p.shape:
        raise ValueError("snp_mat_panle and snp_mat_panle_quality_p must have the same shape.")
    if snp_mat_panle.shape[1] != panel_all.shape[1]:
        raise ValueError("The number of columns in snp_mat_panle and panel_all must be the same.")
    if snp_mat_panle.empty or panel_all.empty:
        return np.zeros(shape=(0, 0))

    snp_mat = snp_mat_panle.to_numpy(dtype=np.int8) - 1  # 1/2 映射为 0/1
    quality_p = snp_mat_panle_quality_p.to_numpy(dtype=np.float32)
    panel_all_np = panel_all.to_numpy(dtype=np.int8)

    return _compute_read_panel_similarity(snp_mat, quality_p, panel_all_np)



def get_maodun(snp_mat_panle,idx_q,idx_k):
    reads_p=snp_mat_panle.iloc[idx_q,:].to_numpy()
    reads_k=snp_mat_panle.iloc[idx_k,:].to_numpy()
    bool_q_k=(reads_p!=0)&(reads_k!=0)
    if sum(bool_q_k)==0:
        return 0,0
    reads_p=reads_p[bool_q_k]
    reads_k=reads_k[bool_q_k]
    maodun= sum(reads_p!=reads_k)
    return maodun>0,maodun/len(reads_p)




def get_best_top_hytyplot(bamfile_file,genomic_file,genotype_file,offset_genotype,offset_header,max_index,length_of_temp_panel=100):
    num_rank= 10
    start_time = time.time()
    bamfile=pysam.AlignmentFile(bamfile_file, "rb")
    # i=0
    ix=np.random.randint(0,max_index)
    # time_start = time.time()
    temp_panel,regions=get_temp_panel(genotype_file,offset_genotype,genotype_file.replace('.txt', '_header.txt'),offset_header,ix,length_of_temp_panel)
    # temp_panel=temp_panel.iloc[:,:20]
    chrom,start,end=regions
    ref_fasta = pysam.FastaFile(genomic_file)
    golbal_snp= get_gold_snp(bamfile,chrom,start,end)

    if not golbal_snp:  # 如果字典为空，条件为True
        print("golbal_snp是空字典")
    else:
        print("golbal_snp不是空字典，内容如下：")
        for pos, ref_base in golbal_snp.items():
            print(f"位置: {pos}, 参考碱基: {ref_base}")

    golbal_snp=remove_consecutive_values(golbal_snp,chrom)
    reads_snp_linkage= get_reads_snp_linkage(temp_panel,bamfile,golbal_snp,chrom,start,end)
    snp_mat,snp_mat_quality=get_snp_mat_quality(reads_snp_linkage,golbal_snp,chrom)
    snp_mat,snp_mat_quality,snp_mat_panle,panel_all,panle_snp_list_index,panle_snp_list,snp_mat_panle_quality=get_panel_haptype_snp_mat_quality(snp_mat,snp_mat_quality,temp_panel)

    snp_mat_panle_quality_p = 10**((-1)*snp_mat_panle_quality/10)
    reads_p = get_reads_p_for_every_k(snp_mat_panle,snp_mat_panle_quality_p,panel_all)
    print(reads_p.shape[0])
    sorted_reads_p = (-1*reads_p).argsort(axis=1)[:,:num_rank]
    sorted_reads_p_flatten = sorted_reads_p.reshape(1,-1)[0,:].tolist()
    condidata = sorted(set(sorted_reads_p_flatten))
    panel_oder = []
    panel_oder_f  = []
    for  i_ in condidata:
        panel_oder.append(i_)
        panel_oder_f.append(sorted_reads_p_flatten.count(i_))
    panel_oder_and_f= np.array([panel_oder,panel_oder_f])[:,np.argsort(panel_oder_f)[::-1]]
    end_time = time.time()
    print('time',end_time-start_time)
    return panel_oder_and_f[0][:5].tolist()

def get_max_index(file_path):
    max_index=0
    with open(file_path, 'r') as f:
        for line in f:
            max_index+=1
    return max_index

# 是整条染色体上最相近的50个单倍型
def get_best_top_hytyplot_in_one_chrmosm(bamfile_file,genomic_file,genotype_file,offset_genotype,offset_header,length_of_temp_panel=100,times=20):
    panel_use_list=[]
    max_index=get_max_index(genotype_file.replace('.txt', '_header.txt'))
    max_index=max_index- length_of_temp_panel
    for i in range(times):
        # print(i)
        panel_use_list.extend(get_best_top_hytyplot(bamfile_file,genomic_file,genotype_file,offset_genotype,offset_header,max_index,length_of_temp_panel))
    # 使用 Counter 统计每个元素的出现次数
    counter = Counter(panel_use_list)
    # 按照出现频率排序
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # sorted_items[:50,0]
    panel_use_list_index=[]
    for i in range(min([50,len(sorted_items)])):
        panel_use_list_index.append(sorted_items[i][0])
    return panel_use_list_index


def get_phase(bamfile_file,genomic_file,genotype_file,offset_genotype,offset_header,panel_use_list_index,start_index,length_of_temp_panel=100):
    #获取bamfile_file 是bam的 path
    #genomic_file 是genomic的 path
    #panel_use_list_index 是整条染色体上最相近的50个单倍型   是get_best_top_hytyplot_in_one_chrmosm的输出结果
    #start_index 是染色体上开始的位置
    #length_of_temp_panel 是单倍型上要使用的长度
    #返回值是phase的结果
    num_rank= 10
    threshold=1
    threshold_reads_p=0.1

    phasing_temp_snp_one_region=[]

    start_time = time.time()
    bamfile=pysam.AlignmentFile(bamfile_file, "rb")

    temp_panel,regions=get_temp_panel(genotype_file,offset_genotype,genotype_file.replace('.txt', '_header.txt'),offset_header,start_index,length_of_temp_panel)
    temp_panel=temp_panel.iloc[panel_use_list_index,:]
    # temp_panel=temp_panel.iloc[:20,:]
    chrom,start,end=regions
    ref_fasta = pysam.FastaFile(genomic_file)
    golbal_snp= get_gold_snp(bamfile,chrom,start,end)
    golbal_snp=remove_consecutive_values(golbal_snp,chrom)
    reads_snp_linkage= get_reads_snp_linkage(temp_panel,bamfile,golbal_snp,chrom,start,end)
    snp_mat,snp_mat_quality=get_snp_mat_quality(reads_snp_linkage,golbal_snp,chrom)
    snp_mat,snp_mat_quality,snp_mat_panle,panel_all,panle_snp_list_index,panle_snp_list,snp_mat_panle_quality=get_panel_haptype_snp_mat_quality(snp_mat,snp_mat_quality,temp_panel)

    overlap_group=get_overlap_group(snp_mat_panle)
    snp_mat_panle_quality_p= 10**((-1)*snp_mat_panle_quality/10)
    reads_p = get_reads_p_for_every_k(snp_mat_panle,snp_mat_panle_quality_p,panel_all)
    print(reads_p.shape[0])
    sorted_reads_p=(-1*reads_p).argsort(axis=1)[:,:num_rank]
    sorted_reads_p_flatten= sorted_reads_p.reshape(1,-1)[0,:].tolist()
    condidata= sorted(set(sorted_reads_p_flatten))
    panel_oder=[]

    
    panel_oder_f=[]
    for  i_ in condidata:
        panel_oder.append(i_)
        panel_oder_f.append(sorted_reads_p_flatten.count(i_))
    panel_oder_and_f= np.array([panel_oder,panel_oder_f])[:,np.argsort(panel_oder_f)[::-1]]
    covage_reads=[]
    old_coverage_reads=[]
    mem=[]
    mem_temp=[]
    usefull_i_=[]
    all_hytyplot_i = None
    #第一轮过滤
    all_temp_i=[]
    all_temp_i_sort=[]
    for i_ in range(panel_oder_and_f.shape[1]):
        temp_i_= np.arange(sorted_reads_p.shape[0])[(sorted_reads_p==panel_oder_and_f[0,i_] ).sum(axis=1)>0].tolist()
        covage_reads.extend(temp_i_)

        mem_i=(len(set(covage_reads))-len(set(old_coverage_reads)))/reads_p.shape[0]
        if mem_i>0.15:
            usefull_i_.append(i_)
            all_temp_i.append(temp_i_)
        mem.append(mem_i)
        mem_temp.append(len(temp_i_))
        if (len(set(covage_reads))>reads_p.shape[0]*0.95) and (i_>=1):

            mem_and_mem_temp= np.round(np.vstack([np.round(np.array(mem),4),np.array(mem_temp)]),4)

            all_hytyplot_i = np.vstack([panel_oder_and_f[0,usefull_i_].astype(int),mem_and_mem_temp[0,usefull_i_]])
            all_hytyplot_i_argsort=(-1*(all_hytyplot_i[1,:])).argsort()
            all_hytyplot_i = all_hytyplot_i[:,all_hytyplot_i_argsort]
            for i__ in all_hytyplot_i_argsort:
                all_temp_i_sort.append(all_temp_i[i__])

            break
        old_coverage_reads=covage_reads.copy()
    #第二轮过滤
    if all_hytyplot_i is not None and all_hytyplot_i.shape[1] > 2:
        reads_map_num_diff=[]
        old_all_temp_i_sort=[]
        for i_ in range(0,all_hytyplot_i.shape[1]):
            reads_map_num_diff.append(len(set(all_temp_i_sort[i_])-set(old_all_temp_i_sort)))
            old_all_temp_i_sort.extend(all_temp_i_sort[i_])
        reads_map_num_diff = np.array(reads_map_num_diff)
        if len(reads_map_num_diff)>2:
            bool_threshold = reads_map_num_diff>threshold
            reads_map_num_diff = reads_map_num_diff[bool_threshold]
        # print(len(reads_map_num_diff))
        all_hytyplot_i = all_hytyplot_i[:,bool_threshold]
        all_hytyplot_i = all_hytyplot_i[:,np.argsort(reads_map_num_diff)[::-1]]
    print(all_hytyplot_i)
    #开始分相

    # 第二轮过滤前先检查 all_hytyplot_i 是否有效
    if all_hytyplot_i is not None and all_hytyplot_i.size > 0:
        # 有效时：使用筛选后的单倍型和概率数据
        choosed_hytyplot_i = panel_all.iloc[list(map(int, all_hytyplot_i[0,:])), :].to_numpy()
        choosed_reads_p = reads_p[:, list(map(int, all_hytyplot_i[0,:]))]
    else:
        # 无效时：不筛选，使用全部原始数据
        choosed_hytyplot_i = panel_all.to_numpy()  # 全部单倍型转换为数组
        choosed_reads_p = reads_p.copy()  # 全部reads匹配概率矩阵




    if choosed_hytyplot_i.shape[0]==2:
        phasing_temp=choosed_reads_p.argmax(axis=1)
        choosed_reads_p_max_value=choosed_reads_p.max(axis=1)
        # bool_threshold_reads_p=choosed_reads_p_max_value>threshold_reads_p
        fix_p=choosed_reads_p_max_value/choosed_reads_p.sum(axis=1)
        bool_threshold_reads_p=fix_p<threshold_reads_p
        phasing_temp[bool_threshold_reads_p]=-1
        phasing_0= snp_mat_panle.iloc[phasing_temp==0,:].to_numpy()
        phasing_1= snp_mat_panle.iloc[phasing_temp==1,:].to_numpy()
    elif choosed_hytyplot_i.shape[0]>2:
        # print("---")
        phasing_temp=choosed_reads_p.argmax(axis=1)

        for idx,v in enumerate(phasing_temp):
            if v > 1 :
                for group in overlap_group:
                    if idx in group:
                        len_group=len(group)
                        if len_group==1:
                            bigger= np.argmax(choosed_reads_p[idx,[0,1]])
                            if bigger in [0,1]:
                                phasing_temp[idx]=bigger
                        else:
                            reads_phsing_id_in_group=phasing_temp[group]
                            reads_phsing_id_in_group_bool=np.isin(reads_phsing_id_in_group,[0,1])
                            reads_phsing_id_in_group=reads_phsing_id_in_group[reads_phsing_id_in_group_bool]
                            reads_id_in_group=np.array(group)[reads_phsing_id_in_group_bool]
                            if len(reads_id_in_group)>=1:
                                maoduns=[]
                                maoduns_0=[]
                                maoduns_1=[]
                                maodun_rates=[]
                                maodun_rates_0=[]
                                maodun_rates_1=[]
                                for _,idx_ in  enumerate(reads_id_in_group):
                                    maodun,maodun_rate=get_maodun(snp_mat_panle,idx_,idx)
                                    maoduns.append(maodun)
                                    maodun_rates.append(maodun_rate)
                                    if reads_phsing_id_in_group[_]==0:
                                        maoduns_0.append(maodun)
                                        maodun_rates_0.append(maodun_rate)
                                    elif reads_phsing_id_in_group[_]==1:
                                        maoduns_1.append(maodun)
                                        maodun_rates_1.append(maodun_rate)

                                if sum(maoduns)==0:
                                    bigger= np.argmax(choosed_reads_p[idx,[0,1]])
                                    if bigger in [0,1]:
                                        phasing_temp[idx]=bigger
                                else:
                                    maoduns_bool_in_all_group_phsing = reads_phsing_id_in_group[maoduns]
                                    if  len(set(maoduns_bool_in_all_group_phsing))==1:
                                        if list(set(maoduns_bool_in_all_group_phsing))[0]==0:
                                            phasing_temp[idx]=1
                                        elif list(set(maoduns_bool_in_all_group_phsing))[0]==1:
                                            phasing_temp[idx]=0
                                    elif len(set(maoduns_bool_in_all_group_phsing))>1:
                                        if sum(maodun_rates_0)>=sum(maodun_rates_1):
                                            phasing_temp[idx]=1
                                        else:
                                            phasing_temp[idx]=0

    else:
        phasing_temp=np.random.randint(0,2,choosed_reads_p.shape[0])


    # 先检查 phasing_temp 是否为空
    if len(phasing_temp) == 0:
        # 处理空列表的情况，例如直接跳过判断或返回默认值
        # 示例：跳过当前逻辑，继续后续代码
        snp_mat_panle_0=snp_mat_panle.to_numpy()
        snp_mat_panle_1=snp_mat_panle.to_numpy()
        phasing_temp_0=[]
    else:
        # 进行具体的分相
        if (sum(phasing_temp==0)/len(phasing_temp)<0.1) or (sum(phasing_temp==1)/len(phasing_temp)<0.1):
            old_phasing_temp=phasing_temp
            old_phasing_temp_bool=old_phasing_temp==1
            phasing_temp=np.random.randint(0,2,choosed_reads_p.shape[0])
            phasing_temp[old_phasing_temp_bool]=1
        snp_mat_panle_0=snp_mat_panle.iloc[phasing_temp==0,:].to_numpy()
        snp_mat_panle_1=snp_mat_panle.iloc[phasing_temp==1,:].to_numpy()
        phasing_temp_0=[]
    for col in range(snp_mat_panle_0.shape[1]):
        col_value= list(set(snp_mat_panle_0[:,col])-set([0]))
        if len(col_value)==1:
            phasing_temp_0.append(col_value[0])
        elif len(col_value)==0:
            phasing_temp_0.append(0)
        elif len(col_value)==2:
            phasing_temp_0.append(2)
        else:
            print("some thing wrong")
    phasing_temp_1=[]
    for col in range(snp_mat_panle_1.shape[1]):
        col_value= list(set(snp_mat_panle_1[:,col])-set([0]))
        if len(col_value)==1:
            phasing_temp_1.append(col_value[0])
        elif len(col_value)==0:
            phasing_temp_1.append(0)
        elif len(col_value)==2:
            phasing_temp_1.append(2)
    phasing_temp_snp= np.vstack([phasing_temp_0,phasing_temp_1])
    phasing_temp_snp_one_region.append(phasing_temp_snp)
    #phasing_temp_snp_one_region = pd.DataFrame(np.vstack(phasing_temp_snp_one_region))
    #phasing_temp_snp_one_region.columns=snp_mat_panle.columns
    end_time=time.time()
    print("phasing time",end_time-start_time)
    return phasing_temp_snp_one_region,snp_mat_panle

def rename_columns_with_full_info(df, vcf_to_csv):
    """
    将DataFrame的列名从CHR_POS格式转换为CHROM:POS_REF_ALT格式
    
    Parameters:
    df: 要修改列名的DataFrame
    vcf_to_csv: VCF转换后的txt文件路径
    
    Returns:
    修改列名后的DataFrame
    """
    import tempfile
    import os
    
    # 获取原始VCF文件路径
    vcf_file = vcf_to_csv.replace('.txt', '.vcf')
    
    # 在临时目录中创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, dir=os.path.dirname(vcf_to_csv)) as tmp_genotype, \
         tempfile.NamedTemporaryFile(suffix='_header.txt', delete=False, dir=os.path.dirname(vcf_to_csv)) as tmp_header:
        tmp_genotype_path = tmp_genotype.name
        tmp_header_path = tmp_header.name
    
    try:
        # 使用fast_vcf_to_txt_full生成临时的完整header文件
        fast_vcf_to_txt_full(vcf_file, tmp_genotype_path.replace('.txt', ''))
        
        # 构建旧列名（CHR_POS）到新列名（CHROM:POS_REF_ALT）的映射
        col_name_mapping = {}
        
        # 从完整的header文件中读取信息
        with open(tmp_header_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
                    old_col_name = f"{chrom}_{pos}"
                    new_col_name = f"{chrom}:{pos}_{ref}_{alt}"
                    col_name_mapping[old_col_name] = new_col_name
        
        # 检查df的列名是否在映射中
        df_cols = [col for col in df.columns if col != 'bam_file']
        matching_cols = [col for col in df_cols if col in col_name_mapping]
        
        print(f"找到 {len(matching_cols)} 个匹配的列名")
        
        # 如果有不匹配的列，显示前几个作为示例
        missing_cols = [col for col in df_cols if col not in col_name_mapping]
        if missing_cols:
            print(f"警告: 有 {len(missing_cols)} 个列名在header文件中找不到映射")
            print(f"例如: {missing_cols[:3]}")
        
        # 替换df的列名
        df.rename(columns=col_name_mapping, inplace=True)
        
    finally:
        # 删除临时文件
        try:
            os.remove(tmp_genotype_path)
 #           os.remove(tmp_header_path)
        except:
            pass
    
    return df

def process_single_bam(bam_file, genomic_file, vcf_to_csv, offset_genotype, offset_header, length_of_temp_panel):   


    bam_path = os.path.join(BAM_DIR, bam_file)
    # 获取分相结果
    panel_use_list_index = get_best_top_hytyplot_in_one_chrmosm(
        bam_path, genomic_file, vcf_to_csv, offset_genotype, offset_header,length_of_temp_panel=300,times=20)
    
    phasing_results, snp_mat_panle = get_phase(
        bam_path, genomic_file, vcf_to_csv, offset_genotype, offset_header,
        panel_use_list_index,0,length_of_temp_panel=length_of_temp_panel
    )

    # 转换为DataFrame并添加文件名标识
    df = pd.DataFrame(np.vstack(phasing_results), columns=snp_mat_panle.columns)
    df.insert(0, 'bam_file', bam_path)  # 在第一列添加文件名

    return df



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True, help="Input VCF file")
    parser.add_argument("--bam-dir", required=True, help="Directory containing BAM files")
    parser.add_argument("--genome", required=True, help="Genome reference file")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()




if __name__ == '__main__':
    args = parse_args()
    global BAM_DIR
    BAM_DIR = args.bam_dir
    # 原 main() 中针对单个 VCF 的逻辑
    vcf_to_csv = args.vcf.replace('.vcf', '.txt')
    fast_vcf_to_txt(args.vcf, vcf_to_csv)
    offset_genotype = build_line_index(vcf_to_csv)
    offset_header = build_line_index(vcf_to_csv.replace('.txt', '_header.txt'))
    
    with open(vcf_to_csv.replace('.txt', '_header.txt'), 'r') as f:
        length_of_temp_panel = sum(1 for _ in f)
    
    all_bam_files = sorted([f for f in os.listdir(args.bam_dir) if f.endswith('.bam')])
    

    with Pool(processes=32) as pool:
        process_func = functools.partial(
            process_single_bam,
            genomic_file=args.genome,
            vcf_to_csv=vcf_to_csv,
            offset_genotype=offset_genotype,
            offset_header=offset_header,
            length_of_temp_panel=length_of_temp_panel
        )
        results = pool.map(process_func, all_bam_files)
    


    pd.concat(results).to_csv(args.output, index=False)



