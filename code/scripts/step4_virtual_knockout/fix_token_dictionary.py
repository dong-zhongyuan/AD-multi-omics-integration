#!/usr/bin/env python3
"""
修复 token 字典：将特殊 token 的 ID 设置在模型词汇表范围内
"""

import pickle
from pathlib import Path

# 路径
token_dict_path = Path("tools/geneformer-main/geneformer/token_dictionary_gc104M.pkl")
backup_path = token_dict_path.with_suffix('.pkl.backup')

# 模型词汇表大小
MODEL_VOCAB_SIZE = 20275  # token ID 范围: 0-20274

print("=" * 80)
print("修复 Token 字典")
print("=" * 80)

# 1. 备份原始文件
print(f"\n[1/4] 备份原始文件...")
if not backup_path.exists():
    import shutil
    shutil.copy(token_dict_path, backup_path)
    print(f"  ✅ 备份到: {backup_path}")
else:
    print(f"  ⚠️  备份已存在，跳过")

# 2. 加载原始 token 字典
print(f"\n[2/4] 加载原始 token 字典...")
with open(token_dict_path, "rb") as f:
    old_token_dict = pickle.load(f)
print(f"  原始大小: {len(old_token_dict)}")

# 3. 创建新的 token 字典
print(f"\n[3/4] 创建新的 token 字典...")
new_token_dict = {}

# 过滤基因：只保留 token_id < MODEL_VOCAB_SIZE - 2（为特殊 token 预留空间）
max_gene_id = MODEL_VOCAB_SIZE - 3  # 预留 20273, 20274 给特殊 token
gene_count = 0

for gene, token_id in old_token_dict.items():
    # 跳过旧的特殊 token
    if gene in ['<cls>', '<eos>', '<pad>', '<mask>']:
        continue
    
    # 只保留在有效范围内的基因
    if token_id <= max_gene_id:
        new_token_dict[gene] = token_id
        gene_count += 1

# 添加特殊 token（放在词汇表末尾）
new_token_dict['<cls>'] = MODEL_VOCAB_SIZE - 2  # 20273
new_token_dict['<eos>'] = MODEL_VOCAB_SIZE - 1  # 20274

print(f"  保留基因数量: {gene_count}")
print(f"  特殊 token:")
print(f"    <cls>: {new_token_dict['<cls>']}")
print(f"    <eos>: {new_token_dict['<eos>']}")
print(f"  新字典大小: {len(new_token_dict)}")

# 检查共识基因是否保留
# 注意：这些基因用于验证 token 字典修复是否成功，类似单元测试中的 sanity check
# 来源：step3 转录组 brain_hubs 中的高分基因（工具脚本，不影响主管线结果）
consensus_genes = ['JUNB', 'THUMPD1', 'PALB2', 'MT-CYB', 'FTH1']
print(f"\n  共识基因检查:")
all_present = True
for gene in consensus_genes:
    if gene in new_token_dict:
        print(f"    ✅ {gene}: {new_token_dict[gene]}")
    else:
        print(f"    ❌ {gene}: 不在字典中")
        all_present = False

if not all_present:
    print("\n❌ 错误：部分共识基因不在新字典中！")
    exit(1)

# 4. 保存新的 token 字典
print(f"\n[4/4] 保存新的 token 字典...")
with open(token_dict_path, "wb") as f:
    pickle.dump(new_token_dict, f)
print(f"  ✅ 保存到: {token_dict_path}")

print("\n" + "=" * 80)
print("✅ Token 字典修复完成！")
print("=" * 80)
print(f"原始大小: {len(old_token_dict)} → 新大小: {len(new_token_dict)}")
print(f"最大 token ID: {max(new_token_dict.values())} (模型支持: 0-{MODEL_VOCAB_SIZE-1})")
