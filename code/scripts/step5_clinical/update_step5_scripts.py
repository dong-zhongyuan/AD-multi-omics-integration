#!/usr/bin/env python3
"""
Step 5 临床验证脚本更新工具

功能：
1. 从Tier 1候选基因文件读取基因列表
2. 更新所有Step 5脚本中的硬编码基因列表
3. 删除旧的分析结果
4. 生成更新报告

更新的脚本：
- step5_survival/prepare_adni_survival_data.py
- step5_survival/prepare_adni_survival_data.R
- step5_survival/run_cox_survival_analysis.R
- step5_survival/run_cox_survival_analysis_simple.R
- step5_nhanes/run_nhanes_optimized_2017.R
- step5_gbd/run_gbd_analysis.py
"""

import sys
sys.path.insert(0, "")
from tools.config_loader import get_config
config = get_config()

import pandas as pd
import re
from pathlib import Path
import shutil
from datetime import datetime

# 配置
TIER1_FILE = "output/step4_virtual_knockout/clinical_candidates/tier1_clinical_candidates.csv"
SCRIPTS_DIR = Path("scripts/step5_clinical")
OUTPUT_DIR = Path("output/step5_clinical")
BACKUP_DIR = Path("output/step5_clinical_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

def load_tier1_genes():
    """加载Tier 1候选基因"""
    df = pd.read_csv(TIER1_FILE)
    genes = df['Gene'].tolist()
    print(f"✓ 加载Tier 1基因: {len(genes)} 个")
    print(f"  {', '.join(genes)}")
    return genes

def backup_old_results():
    """备份旧的分析结果"""
    if OUTPUT_DIR.exists():
        print(f"\n✓ 备份旧结果到: {BACKUP_DIR}")
        shutil.copytree(OUTPUT_DIR, BACKUP_DIR)
        shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True)
    else:
        print("\n✓ 没有旧结果需要备份")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def update_survival_python(genes):
    """更新 prepare_adni_survival_data.py"""
    script_path = SCRIPTS_DIR / "step5_survival/prepare_adni_survival_data.py"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 生成新的基因字典（需要probe映射，这里先用占位符）
    gene_dict_str = "candidate_genes = {\n"
    for gene in genes:
        gene_dict_str += f'    "{gene}": [],  # TODO: 需要添加probe ID\n'
    gene_dict_str += "}"
    
    # 替换硬编码的基因字典
    content = re.sub(
        r'candidate_genes\s*=\s*\{[^}]+\}',
        gene_dict_str,
        content,
        flags=re.DOTALL
    )
    
    # 替换基因表达字典
    expr_dict_lines = []
    for gene in genes:
        expr_dict_lines.append(f'        "{gene}": gene_expr_dict.get("{gene}", np.nan),')
    expr_dict_str = '\n'.join(expr_dict_lines)
    
    content = re.sub(
        r'("IL7R":\s*gene_expr_dict\.get.*?"CLU":\s*gene_expr_dict\.get[^}]+)',
        expr_dict_str,
        content,
        flags=re.DOTALL
    )
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 更新: {script_path.name}")
    print(f"  ⚠️  需要手动添加probe ID映射")

def update_survival_r(genes):
    """更新 prepare_adni_survival_data.R"""
    script_path = SCRIPTS_DIR / "step5_survival/prepare_adni_survival_data.R"
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 生成新的基因列表
    gene_list_str = "candidate_genes <- list(\n"
    for gene in genes:
        gene_list_str += f'  {gene} = c(),  # TODO: 需要添加probe ID\n'
    gene_list_str += ")"
    
    # 替换硬编码的基因列表
    content = re.sub(
        r'candidate_genes\s*<-\s*list\([^)]+\)',
        gene_list_str,
        content,
        flags=re.DOTALL
    )
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ 更新: {script_path.name}")
    print(f"  ⚠️  需要手动添加probe ID映射")

def update_cox_analysis_r(genes):
    """更新 run_cox_survival_analysis.R 和 run_cox_survival_analysis_simple.R"""
    
    for script_name in ["run_cox_survival_analysis.R", "run_cox_survival_analysis_simple.R"]:
        script_path = SCRIPTS_DIR / f"step5_survival/{script_name}"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 生成新的基因向量
        genes_str = ', '.join([f'"{g}"' for g in genes])
        new_line = f'candidate_genes <- c({genes_str})'
        
        # 替换硬编码的基因列表
        content = re.sub(
            r'candidate_genes\s*<-\s*c\([^)]+\)',
            new_line,
            content
        )
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ 更新: {script_name}")

def update_nhanes_r(genes):
    """更新 run_nhanes_optimized_2017.R"""
    script_path = SCRIPTS_DIR / "step5_nhanes/run_nhanes_optimized_2017.R"
    
    print(f"⚠️  {script_path.name} 需要手动更新")
    print(f"   原因：NHANES脚本需要基因到生物标志物的映射")
    print(f"   建议：保留原有的5个基因（IL7R, LTB, JUNB, MT-CYB, FTH1）")
    print(f"   或者为新基因添加对应的生物标志物映射")

def update_gbd_python(genes):
    """更新 run_gbd_analysis.py"""
    script_path = SCRIPTS_DIR / "step5_gbd/run_gbd_analysis.py"
    
    print(f"⚠️  {script_path.name} 需要手动更新")
    print(f"   原因：GBD脚本中的基因描述是硬编码的")
    print(f"   建议：更新基因描述部分（第224-240行）")

def generate_gene_config_file(genes):
    """生成基因配置文件供脚本读取"""
    config_file = SCRIPTS_DIR / "tier1_genes_config.txt"
    
    with open(config_file, 'w') as f:
        for gene in genes:
            f.write(f"{gene}\n")
    
    print(f"\n✓ 生成基因配置文件: {config_file}")
    print(f"  脚本可以从此文件动态读取基因列表")

def generate_update_report(genes):
    """生成更新报告"""
    report_file = SCRIPTS_DIR / "STEP5_UPDATE_REPORT.md"
    
    report = []
    report.append("# Step 5 脚本更新报告\n")
    report.append(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## 更新的基因列表\n")
    report.append(f"从 {len(['IL7R', 'LTB', 'JUNB', 'FTH1', 'CLU'])} 个基因更新到 {len(genes)} 个基因：\n")
    for i, gene in enumerate(genes, 1):
        report.append(f"{i}. {gene}")
    
    report.append("\n## 已自动更新的脚本\n")
    report.append("- ✅ `step5_survival/run_cox_survival_analysis.R`")
    report.append("- ✅ `step5_survival/run_cox_survival_analysis_simple.R`")
    
    report.append("\n## 需要手动更新的脚本\n")
    report.append("### 1. `step5_survival/prepare_adni_survival_data.py`")
    report.append("- **原因**: 需要为每个基因添加Affymetrix probe ID映射")
    report.append("- **操作**: 在ADNI数据字典中查找每个基因对应的probe ID")
    report.append("- **示例**:")
    report.append("  ```python")
    report.append('  candidate_genes = {')
    report.append('      "CHMP5": ["PROBE_ID_1", "PROBE_ID_2"],  # 需要查找')
    report.append('      "THUMPD1": ["PROBE_ID_3"],')
    report.append('      ...')
    report.append('  }')
    report.append("  ```")
    
    report.append("\n### 2. `step5_survival/prepare_adni_survival_data.R`")
    report.append("- **原因**: 同上，需要probe ID映射")
    
    report.append("\n### 3. `step5_nhanes/run_nhanes_optimized_2017.R`")
    report.append("- **原因**: 需要为每个基因指定对应的NHANES生物标志物")
    report.append("- **建议**: ")
    report.append("  - 保留原有的5个基因（IL7R, LTB, JUNB, MT-CYB, FTH1）")
    report.append("  - 或者为新基因添加生物标志物映射（需要生物学知识）")
    report.append("- **映射示例**:")
    report.append("  - IL7R → 淋巴细胞百分比")
    report.append("  - LTB → CRP + 白细胞计数")
    report.append("  - FTH1 → 铁蛋白")
    
    report.append("\n### 4. `step5_gbd/run_gbd_analysis.py`")
    report.append("- **原因**: 基因功能描述是硬编码的")
    report.append("- **操作**: 更新第224-240行的基因描述")
    
    report.append("\n## 旧结果备份\n")
    if BACKUP_DIR.exists():
        report.append(f"- 备份位置: `{BACKUP_DIR}`")
    else:
        report.append("- 无旧结果需要备份")
    
    report.append("\n## 下一步操作\n")
    report.append("1. 手动更新上述需要手动更新的脚本")
    report.append("2. 运行生存分析:")
    report.append("   ```bash")
    report.append("   cd scripts/step5_clinical/step5_survival")
    report.append("   Rscript run_cox_survival_analysis_simple.R")
    report.append("   ```")
    report.append("3. 运行NHANES分析（如果已更新）:")
    report.append("   ```bash")
    report.append("   cd scripts/step5_clinical/step5_nhanes")
    report.append("   Rscript run_nhanes_optimized_2017.R")
    report.append("   ```")
    report.append("4. 运行GBD分析（如果已更新）:")
    report.append("   ```bash")
    report.append("   cd scripts/step5_clinical/step5_gbd")
    report.append("   python3 run_gbd_analysis.py")
    report.append("   ```")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✓ 生成更新报告: {report_file}")

def main():
    print("=" * 60)
    print("Step 5 临床验证脚本更新")
    print("=" * 60)
    
    # 1. 加载Tier 1基因
    genes = load_tier1_genes()
    
    # 2. 备份旧结果
    backup_old_results()
    
    # 3. 更新脚本
    print("\n更新脚本...")
    update_cox_analysis_r(genes)
    update_survival_python(genes)
    update_survival_r(genes)
    update_nhanes_r(genes)
    update_gbd_python(genes)
    
    # 4. 生成配置文件
    generate_gene_config_file(genes)
    
    # 5. 生成报告
    generate_update_report(genes)
    
    print("\n" + "=" * 60)
    print("✓ 更新完成！")
    print("=" * 60)
    print("\n⚠️  注意：部分脚本需要手动更新")
    print("   请查看: scripts/step5_clinical/STEP5_UPDATE_REPORT.md")

if __name__ == "__main__":
    main()
