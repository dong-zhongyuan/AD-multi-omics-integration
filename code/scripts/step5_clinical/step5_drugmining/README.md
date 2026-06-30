# Drug Mining for Therapeutic Targets

## 📋 修改说明

### 修改内容

1. **动态读取治疗靶点**
   - 原路径: `step2_wgcna/result/candidate_genes_top100.csv`
   - 新路径: `output/step5_gene_classification/therapeutic_targets.csv`
   - 自动读取11个治疗靶点基因（9个蛋白质组学 + 2个转录组学）

2. **输出目录修正**
   - 原路径: `./drug_mining_out`
   - 新路径: `./output/step5_clinical_validation/drug_mining`

### 治疗靶点列表

**蛋白质组学治疗靶点（9个）**:
- CST3, NPTXR, IL16, IGF1R, PSEN1, MAPT, BACE1, AGRN, TREM1

**转录组学治疗靶点（2个）**:
- IP6K1, TRMT44

### 使用方法

#### 方法1: Jupyter Notebook
```bash
# 在Jupyter中打开并运行
jupyter notebook scripts/step5_clinical/step5_drugmining/drugmining.ipynb
```

#### 方法2: 命令行
```bash
# 使用默认路径
python scripts/step5_clinical/step5_drugmining/drugmining.ipynb

# 或指定自定义路径
python scripts/step5_clinical/step5_drugmining/drugmining.ipynb \
  --gene_list output/step5_gene_classification/therapeutic_targets.csv \
  --out_dir output/step5_clinical_validation/drug_mining
```

### 输出文件

所有输出保存在 `output/step5_clinical_validation/drug_mining/`:

1. **drug_mining_all.csv** - 所有基因的完整药物挖掘结果
2. **drug_mining_ranked.csv** - 按DrugEvidenceScore排序的结果
3. **drug_mining_summary.json** - 汇总统计信息

### 数据来源

脚本整合以下数据库：

1. **Open Targets** - 药物靶点关联、临床试验阶段
2. **DGIdb** - 药物-基因相互作用
3. **ChEMBL** - 化合物活性数据
4. **HGNC** - 基因命名标准化

### 评分系统

**DrugEvidenceScore** 综合考虑：
- Open Targets关联评分
- 已批准药物数量
- 临床试验最高阶段
- 靶点可成药性评分
- DGIdb交互数量
- ChEMBL活性数据

### 预期结果

根据历史运行结果，预期发现：
- **ApprovedTarget**: 已有FDA批准药物的靶点
- **LateClinicalTarget**: 处于III期临床试验的靶点
- **EarlyClinicalTarget**: 处于I/II期临床试验的靶点
- **Tractable**: 具有可成药性但尚无临床药物的靶点

### 注意事项

1. **网络依赖**: 需要访问外部API（Open Targets, DGIdb, ChEMBL）
2. **运行时间**: 11个基因预计需要5-10分钟
3. **缓存机制**: 使用`drug_mining_cache/`目录缓存API结果
4. **速率限制**: 自动处理API速率限制，失败时会重试

---

修改时间: 2026-05-16
修改者: 狗狗 🐶
