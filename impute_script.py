import pandas as pd
import numpy as np
# import sklearn.model_selection as ms
import os
import gzip
import subprocess
import argparse
from bgzip import BGZipWriter
from tempfile import NamedTemporaryFile
import pysam
import torch
from torch import nn

class my_cnn(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,kernel_size,stride,seq_len,d_model,dropout,device="cuda"):
        super(my_cnn, self).__init__()
        self.position_encoding = nn.Parameter(torch.randn(d_model,seq_len)).to(device)
        # self.position_encoding.to(device)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding="same"
        self.dropout=dropout
        self.conv1=nn.Conv1d(self.input_size,d_model,self.kernel_size,self.stride,self.padding)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(self.dropout)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2=nn.Conv1d(d_model,self.hidden_size,self.kernel_size,self.stride,self.padding)
        self.bn2=nn.BatchNorm1d(self.hidden_size)
        self.bn3=nn.BatchNorm1d(self.hidden_size)
        # self.conv3=nn.Conv1d(self.hidden_size,self.hidden_size,self.kernel_size,self.stride,self.padding)
        # self.conv4=nn.Conv1d(self.hidden_size,self.hidden_size,self.kernel_size,self.stride,self.padding)
        # self.conv5=nn.Conv1d(self.hidden_size,self.hidden_size,self.kernel_size,self.stride,self.padding)
        # self.conv6=nn.Conv1d(self.hidden_size,self.hidden_size,self.kernel_size,self.stride,self.padding)
        # self.conv7=nn.Conv1d(self.hidden_size,self.hidden_size,self.kernel_size,self.stride,self.padding)
        #transformer
        self.transformer1=nn.TransformerEncoderLayer(hidden_size,nhead=1,dropout=self.dropout)
        #self.transformer2=nn.TransformerEncoderLayer(hidden_size,nhead=1,dropout=self.dropout)
        #self.transformer3=nn.TransformerEncoderLayer(hidden_size,nhead=1,dropout=self.dropout)
        # self.transformer4=nn.TransformerEncoderLayer(hidden_size,nhead=1,dropout=self.dropout)
        self.conv8=nn.Conv1d(self.hidden_size,self.output_size,self.kernel_size,self.stride,self.padding)

    def forward(self,x):
        # print(self.position_encoding.shape)
        # print(x.shape)
        x=x.unsqueeze(1)
        x=self.conv1(x)
        x=x+self.position_encoding.squeeze(1)
        x=self.bn1(x)
        # x=x.unsqueeze(1)
        # print(x.shape)
        # x=self.conv1(x)
        x=self.relu(x)
        x=self.drop(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x_=self.drop(x)
        # x=self.conv3(x)
        # x=self.relu(x)
        # x=self.drop(x)
        # x=self.conv4(x)
        # x=self.relu(x)
        # x=self.drop(x)
        # x=self.conv5(x)
        # x=self.relu(x)
        # x=self.drop(x)
        # x=self.conv6(x)
        # x=self.relu(x)
        # x=self.drop(x)
        x=x_.permute(2,0,1)
        x=self.transformer1(x)
        #x=self.relu(x)
        #x=self.transformer2(x)
        #x=self.relu(x)
        #x=self.transformer3(x)
        #x=self.relu(x)
        # x=self.transformer4(x)
        x=x.permute(1,2,0)
        x=self.relu(x)
        x=x+x_
        x=self.bn3(x)
        # x=self.conv7(x)
        # x=self.relu(x)
        # x=self.drop(x)
        x=self.conv8(x)
        return x



from torch.utils.data import Dataset
# class GenotypeDataset(Dataset):
#     # def __init__(self, data,data_un_misss, missing_perc=0.1):
#     def __init__(self, data_,y,miss_perc=0.2):
#         if isinstance(data_, pd.DataFrame):
#             data_ = data_.copy().to_numpy()
#         self.data = data_
#         self.datay = y
#         self.missing_perc=miss_perc
#         # self.bool_miss=bool_miss
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         x = self.data[idx].copy()
#         missing_size = int(self.missing_perc * len(x))
#         missing_index = np.random.randint(len(x), size=missing_size)
#         x[missing_index] = -1
#         y = self.datay[idx]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# class GenotypeDataset(Dataset):
#     # def __init__(self, data,data_un_misss, missing_perc=0.1):
#     def __init__(self, data_,y,miss_num=10):
#         if isinstance(data_, pd.DataFrame):
#             data_ = data_.copy().to_numpy()
#         self.data = data_
#         self.datay = y
#         self.missing_num=miss_num
#         # self.bool_miss=bool_miss
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         x = self.data[idx].copy()
#         missing_index = np.random.randint(len(x), size=self.missing_num)
#         missing_index_len=np.random.randint(5,20,size=self.missing_num)
#         for i in range(self.missing_num):
#             star=max(0,missing_index[i]-missing_index_len[i]//2)
#             end=min(len(x),missing_index[i]+missing_index_len[i]//2)
#             x[star:end] = -1
#         y = self.datay[idx]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class GenotypeDataset(Dataset):
    # def __init__(self, data,data_un_misss, missing_perc=0.1):
    def __init__(self, data_,y,need_imput_data=None,miss_num=10):
        if isinstance(data_, pd.DataFrame):
            data_ = data_.copy().to_numpy()
        self.data = data_
        self.datay = y
        self.missing_num=miss_num
        self.need_imput_data=need_imput_data
        # self.bool_miss=bool_miss
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx].copy()
        if self.need_imput_data is not None:
            # 计算每个特征的缺失概率
            missing_prob = np.mean(self.need_imput_data < 0, axis=0)
            # 按概率生成缺失掩码
            missing_mask = np.random.rand(len(x)) < missing_prob
            x[missing_mask] = -1
        else:
            missing_index = np.random.randint(len(x), size=self.missing_num)
            missing_index_len=np.random.randint(5,20,size=self.missing_num)
            for i in range(self.missing_num):
                star=max(0,missing_index[i]-missing_index_len[i]//2)
                end=min(len(x),missing_index[i]+missing_index_len[i]//2)
                x[star:end] = -1
        y = self.datay[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


#train 函数
def train(model, train_loader, criterion, optimizer, epochs=10, device="cuda"):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs=inputs.float()
            targets=targets.long()

            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to the correct device

            optimizer.zero_grad()  # Reset the gradients

            outputs = model(inputs)  # Forward pass
            # print(outputs.dtype)
            # print(targets.dtype)
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch: {epoch+1}/{epochs}.. Training Loss: {epoch_loss:.3f}')


def to_vcf(y_pre, mask_data, output_vcf_path, sample_ids=None, confidence=None):
    """
    Convert y_pre tensor to VCF format using positions, REF, ALT from mask data and save as .vcf.gz
    
    Args:
        y_pre: Tensor of shape (2*num_individuals, num_sites) containing haplotypes (0/1)
        mask_data: DataFrame containing position information (columns are "chrXX:position_ref_alt")
        output_vcf_path: Path to save the output .vcf.gz file
        sample_ids: List of sample IDs (if None, will extract from mask_data index)
        confidence: Confidence scores array of shape (2*num_individuals, num_sites) (optional)
    """
    # Convert tensor to numpy array if needed
    if isinstance(y_pre, torch.Tensor):
        y_pre = y_pre.cpu().numpy()
    y_pre = y_pre.astype(int)
    
    # Convert confidence to numpy array if provided
    if confidence is not None:
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()
        confidence = confidence.astype(float)
        if confidence.shape != y_pre.shape:
            raise ValueError(f"Confidence array shape {confidence.shape} doesn't match y_pre shape {y_pre.shape}")
    
    # Get dimensions
    num_haps, num_sites = y_pre.shape
    num_individuals = num_haps // 2
    
    # Extract sample IDs from mask_data if not provided
    if sample_ids is None:
        # Get unique sample names by removing _hap1/_hap2 suffixes
        sample_ids = sorted(list(set([idx.split('_')[0] for idx in mask_data.index])))
    
    # Extract chromosome, positions, REF and ALT from mask_data columns
    chrom = None
    positions = []
    refs = []
    alts = []
    
    for col in mask_data.columns:
        # Split column name into components, handling underscores in chromosome names
        # Format should be "chrXX_whatever:position_ref_alt"
     
        # First split on the last underscore to separate alt allele
        last_underscore_pos = col.rfind('_')
        if last_underscore_pos == -1:
            raise ValueError(f"Column name format should be chrXX_whatever:position_ref_alt, got {col}")
        alt = col[last_underscore_pos+1:]
        remaining_part = col[:last_underscore_pos]
    
        # Then split the remaining part on the second-to-last underscore to get ref allele
        prev_underscore_pos = remaining_part.rfind('_')
        if prev_underscore_pos == -1:
            raise ValueError(f"Column name format should be chrXX_whatever:position_ref_alt, got {col}")
        ref = remaining_part[prev_underscore_pos+1:]
        chr_pos_part = remaining_part[:prev_underscore_pos]
    
        # Now split the chromosome:position part
        colon_pos = chr_pos_part.find(':')
        if colon_pos == -1:
            raise ValueError(f"Invalid chromosome:position format in {chr_pos_part}")
    
        chromosome = chr_pos_part[:colon_pos]
        position = chr_pos_part[colon_pos+1:]
    
        # Validate position is numeric
        try:
            position = int(position)
        except ValueError:
            raise ValueError(f"Position should be numeric, got {position} in {col}")
    
        if chrom is None:
            chrom = chromosome  # Get chromosome from first column
        elif chromosome != chrom:
            raise ValueError(f"Inconsistent chromosomes in columns: found {chromosome} and {chrom}")

        positions.append(position)
        refs.append(ref)
        alts.append(alt)

    # Create VCF header
    header = [
        '##fileformat=VCFv4.2',
        f'##contig=<ID={chrom}>',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    ]
    
    # Add confidence FORMAT field definition if confidence is provided
    if confidence is not None:
        header.append('##FORMAT=<ID=CONF,Number=2,Type=Float,Description="Imputation confidence scores for each haplotype">')
    
    header.append('\t'.join(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT'] + sample_ids))
    
    # Prepare genotype data (convert 0/1 to 0|0, 0|1, 1|0, 1|1)
    genotypes = []
    confidences = [] if confidence is not None else None
    
    for i in range(num_individuals):
        hap1 = y_pre[2*i, :]
        hap2 = y_pre[2*i + 1, :]
        genotypes.append([f"{hap1[j]}|{hap2[j]}" for j in range(num_sites)])
        
        if confidence is not None:
            # Get confidence for each haplotype
            conf_hap1 = confidence[2*i, :]
            conf_hap2 = confidence[2*i + 1, :]
            # Store both confidence values separated by comma
            confidences.append([f"{conf_hap1[j]:.4f},{conf_hap2[j]:.4f}" for j in range(num_sites)])
           

    # Transpose genotypes to site-major order
    genotypes = np.array(genotypes).T
    if confidence is not None:
        confidences = np.array(confidences).T
    
    # Create VCF records
    records = []
    for idx, (pos, ref, alt, gt) in enumerate(zip(positions, refs, alts, genotypes)):
        # Prepare INFO field
        info_field = '.'
        
        # Prepare FORMAT field
        format_fields = ['GT']
        if confidence is not None:
            format_fields.append('CONF')
        
        # Prepare sample fields
        sample_fields = []
        if confidence is not None:
            # Get confidence values for this site across all samples
            conf_site = confidences[idx]
            # Combine GT and CONF for each sample
            for g, c in zip(gt, conf_site):
                sample_fields.append(f"{g}:{c}")
        else:
            sample_fields = gt
        
        # VCF fields with real REF/ALT values
        record = [
            chrom,               # CHROM
            str(pos),            # POS
            '.',                 # ID
            ref,                 # REF (from column name)
            alt,                 # ALT (from column name)
            '.',                 # QUAL
            'PASS',              # FILTER
            info_field,          # INFO
            ':'.join(format_fields),  # FORMAT
        ] + sample_fields             # Sample genotypes with confidence
        
        records.append('\t'.join(record))
    


    

    with NamedTemporaryFile(mode='w+', suffix='.vcf') as temp_vcf:
        # Write VCF header
        temp_vcf.write('\n'.join(header) + '\n')
        # Write records
        temp_vcf.write('\n'.join(records) + '\n')
        temp_vcf.flush()  # Ensure all content is written
    
        # Compress temporary file with bgzip from pysam
        pysam.tabix_compress(temp_vcf.name, output_vcf_path, force=True)
    





def calculate_confidence(imputed_sum, key_size, ratio, method='sigmoid', sum_max=None):
    """
    改进版置信度计算：
    - 当imputed_sum < ratio（未达阈值）：计算值取反（1 - 计算值）
    - 当imputed_sum ≥ ratio（超过阈值）：保持原计算值不变
    最终置信度范围限制在[0, 1]
    
    参数:
    - imputed_sum: 求和结果数组
    - key_size: 关键尺寸
    - ratio: 容错阈值
    - method: 计算方法（linear/sigmoid/exponential）
    - sum_max: imputed_sum的最大值（用于映射），若为None则自动计算
    """
    imputed_like = np.zeros(imputed_sum.shape)
    
    # 确定imputed_sum的最大值（用于映射）
    if sum_max is None:
        sum_max = np.max(imputed_sum)
        # 确保sum_max大于ratio，否则使用ratio的2倍作为默认最大值
        if sum_max <= ratio:
            sum_max = ratio * 2
    
    # 区分低于阈值和高于阈值的部分
    mask_below = imputed_sum < ratio
    mask_above = imputed_sum >= ratio

    if method == 'linear':
        # 基础线性计算（未取反）
        linear_below = (imputed_sum[mask_below] / sum_max) * (sum_max / ratio) if ratio != 0 else 0
        linear_above = (imputed_sum[mask_above] - ratio) / (sum_max - ratio) * (1 - ratio/sum_max) + ratio/sum_max
        
        # 未达阈值部分取反，超过阈值部分保持不变
        imputed_like[mask_below] = 1 - linear_below
        imputed_like[mask_above] = linear_above

    elif method == 'sigmoid':
        # 基础Sigmoid计算（未取反）
        scale = 10 / ratio if ratio != 0 else 10
        sigmoid_below = 1 / (1 + np.exp(-scale * (imputed_sum[mask_below] - ratio/2)))
        sigmoid_below = sigmoid_below * (ratio / sum_max)
        
        scale_above = 5 / (sum_max - ratio) if (sum_max - ratio) != 0 else 5
        sigmoid_above = 1 / (1 + np.exp(-scale_above * (imputed_sum[mask_above] - ratio)))
        sigmoid_above = ratio/sum_max + sigmoid_above * (1 - ratio/sum_max)
        
        # 未达阈值部分取反，超过阈值部分保持不变
        imputed_like[mask_below] = 1 - sigmoid_below
        imputed_like[mask_above] = sigmoid_above

    elif method == 'exponential':
        # 基础指数计算（未取反）
        exp_below = (1 - np.exp(-imputed_sum[mask_below] / (ratio * 0.4))) * (ratio / sum_max) if ratio != 0 else 0
        
        above_term = imputed_sum[mask_above] - ratio
        exp_above = ratio/sum_max + (1 - np.exp(-above_term / ((sum_max - ratio) * 0.2))) * (1 - ratio/sum_max)
        
        # 未达阈值部分取反，超过阈值部分保持不变
        imputed_like[mask_below] = 1 - exp_below
        imputed_like[mask_above] = exp_above

    # 确保置信度在[0, 1]范围内
    imputed_like = np.clip(imputed_like, 0, 1)
    return imputed_like


def imput_process_with_confidence(new_data_padded, y_pre, key_size, F_tolerance=0.3):
    ratio = key_size * F_tolerance
    
    if not isinstance(y_pre, np.ndarray):
        y_pre = y_pre.cpu().detach().numpy()
    
    imputed = np.zeros([y_pre.shape[0], y_pre.shape[1] + key_size - 1, key_size])
    
    for i in range(new_data_padded.shape[1] - (key_size - 1)):
        temp_padded = new_data_padded[:, i:i + key_size]
        unique = np.unique(temp_padded, axis=0, return_counts=False)
        index_to_value_map = {i: x for i, x in enumerate(unique)}
        
        i_ = i % key_size
        temp_out = []
        
        for j in y_pre[:, i]:
            try:
                temp_out.append(index_to_value_map[j])
            except:
                temp_out.append(index_to_value_map[0])
        
        temp_out = np.array(temp_out)
        imputed[:, i:i + key_size, i_] = temp_out
    
    imputed_ = imputed[:, (key_size - 1) // 2:-(key_size - 1) // 2]
    imputed_sum = imputed_.sum(axis=2)
    print(imputed_sum)
    # 计算置信度
    confidence = calculate_confidence(imputed_sum, key_size, ratio)
    
    # 生成最终结果（可选，保持原有逻辑）
    imputed_like = np.zeros(imputed_sum.shape)
    imputed_like[imputed_sum >= key_size - ratio] = 1
    imputed_like[imputed_sum <= ratio] = 0
    #imputed_like[((imputed_sum < key_size - ratio) & (imputed_sum > ratio))] = -5
    imputed_like[((imputed_sum < key_size - ratio) & (imputed_sum > ratio))] = 0
    
    return imputed_like, confidence




    # index_data=np.array([value_to_index_map[ tuple(value)] for value in temp_padded])
    # all_index_data.append(index_data[:,None])

def get_accuracy(imputed_like,y):
    return 1-((imputed_like==y)==0).sum()/(imputed_like.shape[0]*imputed_like.shape[1])
def get_accuracy_2(imputed_like,y):
    imputed_like_2= imputed_like[0::2,:]+imputed_like[1::2,:]
    y_2= y[0::2,:]+y[1::2,:]
    return 1-((imputed_like_2==y_2)==0).sum()/(imputed_like_2.shape[0]*imputed_like_2.shape[1])

def get_accuracy_3(imputed_like,y):
    imputed_like_2= imputed_like[0::2,:]+imputed_like[1::2,:]
    y_2= y[0::2,:]+y[1::2,:]
    sum_1= (imputed_like_2<0).sum().sum()

    # return (imputed_like_2==y_2).sum().sum()/((imputed_like_2.shape[0]*imputed_like_2.shape[1])-sum_1//2),sum_1
    return ((imputed_like_2==y_2).sum().sum()+sum_1//2)/((imputed_like_2.shape[0]*imputed_like_2.shape[1])),sum_1

#%%
def impute_again(model,imputed_like):
    if isinstance(imputed_like,np.ndarray):
        imputed_like=torch.tensor(imputed_like)
        imputed_like=imputed_like.float()
        imputed_like=imputed_like.to(device)
        return model(imputed_like).argmax(dim=1)
#%%
# def get_accuracy(imputed_like,y):
#     return 1-((imputed_like==y.numpy())==0).sum()/(imputed_like.shape[0]*imputed_like.shape[1])


def main():
    parser = argparse.ArgumentParser(description='Genotype imputation using CNN')
    parser.add_argument('--train_csv', required=True, help='Training CSV file')
    parser.add_argument('--test_csv', required=True, help='Testing CSV file')
    parser.add_argument('--shapeit_csv', required=True, help='Shapeit CSV file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = pd.read_csv(args.train_csv, index_col=0)
    test_data = pd.read_csv(args.test_csv, index_col=0)
    test_data_shapeit = pd.read_csv(args.shapeit_csv, index_col=0)

    all_data = pd.DataFrame(np.zeros([data.shape[0]*2, int(data.shape[1]/2)]))
    all_data.iloc[0::2,:] = data.iloc[:, 0::2].values
    all_data.iloc[1::2,:] = data.iloc[:,1::2].values
    columns_1 = [i[:-2] for i in data.columns[0::2]]
    all_data.columns = columns_1
    
    all_test = pd.DataFrame(np.zeros([test_data.shape[0]*2, int(test_data.shape[1]/2)]))
    all_test.iloc[0::2,:] = test_data.iloc[:, 0::2].values
    all_test.iloc[1::2,:] = test_data.iloc[:,1::2].values
    columns_1 = [i[:-2] for i in test_data.columns[0::2]]
    all_test.columns = columns_1

    #all_test.columns == test_data_shapeit.columns
    new_data = all_data
    #all_test = test_data
    test_x = all_test

    test_data_shapeit = test_data_shapeit - 1 
    #test_data_shapeit.columns=all_test.columns
    #test_data_shapeit.index = all_test.index     # 替换行名
    test_x_shapeit = test_data_shapeit
    
   


    key_size = 3
    padd_ = np.zeros([new_data.shape[0], (key_size-1)//2]).astype(np.int64)
    new_data_padded = np.concatenate([padd_, new_data, padd_], axis=1)
    
    all_index_data = []
    for i in range(new_data.shape[1]):
        temp_padded = new_data_padded[:, i:i+key_size]
        unique = np.unique(temp_padded, axis=0, return_counts=False)
        value_to_index_map = {tuple(x):i for i,x in enumerate(unique)}
        index_data = np.array([value_to_index_map[tuple(value)] for value in temp_padded])
        all_index_data.append(index_data[:,None])
        
    new_data.isna().sum().sum()
    
    data_y = np.concatenate(all_index_data, axis=1)
    y_max = data_y.max()
    

    model = my_cnn(
        input_size=1,
        hidden_size=256,
        output_size=y_max+1,
        kernel_size=key_size,stride=1,
        seq_len=new_data.shape[1],
        d_model=256,
        dropout=0.0329)




    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000883)
    criterion = nn.CrossEntropyLoss()
    
    if isinstance(test_data_shapeit, pd.DataFrame):
        test_data_shapeit_numpy = test_data_shapeit.to_numpy()
    
    train_dataset = GenotypeDataset(
        data_=new_data,
        y=data_y,
        need_imput_data=test_data_shapeit_numpy,
        miss_num=50
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )
    
    train(
        model,
        train_loader,
        criterion,
        optimizer,
        epochs=200,
        device=device
    )
    
    if isinstance(test_data_shapeit, pd.DataFrame):
        test_data_shapeit = test_data_shapeit.to_numpy()
    
    # Batch processing for inference
    x = torch.Tensor(test_data_shapeit)
    device = next(model.parameters()).device
    predictions = []
    
    batch_size = 16
    for i in range(0, len(x), batch_size):
        batch_x = x[i:i+batch_size].float().to(device)
        
        with torch.no_grad():
            y_pre_batch = model(batch_x).argmax(dim=1)
        
        predictions.append(y_pre_batch.cpu())
    
    y_pre = torch.cat(predictions).numpy()
    
    imputed_like, confidence_scores = imput_process_with_confidence(new_data_padded,y_pre,key_size,F_tolerance=0.5)



    y=all_test
    if isinstance(y, pd.DataFrame):
       y = y.values
    all_haplotype = get_accuracy(imputed_like, y)
    all_bi = get_accuracy_2(imputed_like, y)
    print("填充准确性：" + str(all_bi))
    
    # Save results
    base_name = os.path.basename(args.shapeit_csv).replace("_test.csv", "")
    
    vcf_output_path = os.path.join(
        args.output_dir,
        os.path.basename(args.shapeit_csv).replace("_test.csv", ".imputed_SAC.vcf.gz")
    )
    

    to_vcf(
        y_pre=imputed_like,
        mask_data=test_x_shapeit,
        output_vcf_path=vcf_output_path,
        confidence=confidence_scores  # 从imput_process_with_confidence函数获取的置信度数组
    )




if __name__ == "__main__":
    main()
