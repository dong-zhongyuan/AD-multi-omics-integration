"""
配置加载器
用途: 统一加载和管理项目配置，替代硬编码
作者: 狗狗 🐶
日期: 2026-05-07
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProjectConfig:
    """项目配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，默认为 config/project_config.yaml
        """
        if config_path is None:
            # 自动查找配置文件
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config" / "project_config.yaml"
        
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 解析路径变量
        self._resolve_path_variables()
    
    def _resolve_path_variables(self):
        """解析配置中的路径变量，如 ${project_root}"""
        def resolve_value(value, context):
            if isinstance(value, str) and '${' in value:
                # 替换变量
                for key, val in context.items():
                    if isinstance(val, str):
                        value = value.replace(f'${{{key}}}', val)
            return value
        
        # 首先解析 paths 部分
        if 'paths' in self._config:
            paths = self._config['paths']
            # 多次迭代直到所有变量都被解析
            for _ in range(10):  # 最多10层嵌套
                changed = False
                for key in paths:
                    old_value = paths[key]
                    paths[key] = resolve_value(paths[key], paths)
                    if paths[key] != old_value:
                        changed = True
                if not changed:
                    break
        
        # 解析其他部分的路径
        if 'data_files' in self._config:
            for key in self._config['data_files']:
                self._config['data_files'][key] = resolve_value(
                    self._config['data_files'][key], 
                    self._config.get('paths', {})
                )
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持嵌套键
        
        Args:
            key_path: 配置键路径，如 'paths.project_root' 或 'genes.tier1'
            default: 默认值
        
        Returns:
            配置值
        
        Examples:
            >>> config = ProjectConfig()
            >>> config.get('paths.project_root')
            ''
            >>> config.get('genes.tier1')
            ['CHMP5', 'THUMPD1', ...]
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def get_path(self, key_path: str) -> Path:
        """
        获取路径配置并转换为 Path 对象
        
        Args:
            key_path: 配置键路径
        
        Returns:
            Path 对象
        
        Examples:
            >>> config = ProjectConfig()
            >>> config.get_path('paths.data_dir')
            PosixPath('data')
        """
        path_str = self.get(key_path)
        if path_str is None:
            raise ValueError(f"路径配置不存在: {key_path}")
        return Path(path_str)
    
    def get_genes(self, tier: str = 'tier1') -> List[str]:
        """
        获取指定Tier的基因列表
        
        Args:
            tier: 'tier1', 'tier2', 'tier3', 'all_hub_genes', 'consensus_response_genes'
        
        Returns:
            基因名称列表
        
        Examples:
            >>> config = ProjectConfig()
            >>> config.get_genes('tier1')
            ['CHMP5', 'THUMPD1', 'TMPO', 'TIMM9', 'TOR1AIP1', 'CINP', 'PALB2', 'GADD45GIP1']
        """
        return self.get(f'genes.{tier}', [])
    
    def get_all_candidate_genes(self) -> List[str]:
        """获取所有候选基因（Tier1 + Tier2 + Tier3）"""
        tier1 = self.get_genes('tier1')
        tier2 = self.get_genes('tier2')
        tier3 = self.get_genes('tier3')
        return tier1 + tier2 + tier3
    
    def get_parameter(self, param_path: str, default: Any = None) -> Any:
        """
        获取分析参数
        
        Args:
            param_path: 参数路径，如 'world_model.hidden_dim'
        
        Returns:
            参数值
        
        Examples:
            >>> config = ProjectConfig()
            >>> config.get_parameter('world_model.hidden_dim')
            128
        """
        return self.get(f'parameters.{param_path}', default)
    
    def get_gene_annotation(self, gene: str, field: Optional[str] = None) -> Any:
        """
        获取基因注释信息
        
        Args:
            gene: 基因名称
            field: 注释字段，如 'function', 'pathway'。如果为None，返回所有注释
        
        Returns:
            注释信息
        
        Examples:
            >>> config = ProjectConfig()
            >>> config.get_gene_annotation('TOR1AIP1', 'function')
            '核膜蛋白'
            >>> config.get_gene_annotation('TOR1AIP1')
            {'full_name': '...', 'function': '...', ...}
        """
        annotation = self.get(f'gene_annotations.{gene}')
        if annotation is None:
            return None
        
        if field is None:
            return annotation
        else:
            return annotation.get(field)
    
    def get_gene_biomarkers(self, gene: str) -> List[str]:
        """
        获取基因对应的生物标志物
        
        Args:
            gene: 基因名称
        
        Returns:
            生物标志物列表
        
        Examples:
            >>> config = ProjectConfig()
            >>> config.get_gene_biomarkers('TOR1AIP1')
            ['WBC', 'CRP']
        """
        return self.get(f'gene_biomarker_mapping.{gene}', [])
    
    def get_data_file(self, file_key: str) -> Path:
        """
        获取数据文件路径
        
        Args:
            file_key: 数据文件键，如 'brain_seurat', 'adni_gene_expression'
        
        Returns:
            数据文件路径
        
        Examples:
            >>> config = ProjectConfig()
            >>> config.get_data_file('brain_seurat')
            PosixPath('data/brain-or-csf-raw/brain_seurat.rds')
        """
        file_path = self.get(f'data_files.{file_key}')
        if file_path is None:
            raise ValueError(f"数据文件配置不存在: {file_key}")
        return Path(file_path)
    
    def __repr__(self) -> str:
        return f"ProjectConfig(config_path='{self.config_path}')"


# 全局配置实例（单例模式）
_global_config = None

def get_config(config_path: Optional[str] = None) -> ProjectConfig:
    """
    获取全局配置实例
    
    Args:
        config_path: 配置文件路径，仅在首次调用时有效
    
    Returns:
        ProjectConfig 实例
    
    Examples:
        >>> from tools.config_loader import get_config
        >>> config = get_config()
        >>> tier1_genes = config.get_genes('tier1')
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ProjectConfig(config_path)
    
    return _global_config


# 便捷函数
def get_tier1_genes() -> List[str]:
    """快捷获取Tier1基因列表"""
    return get_config().get_genes('tier1')

def get_project_root() -> Path:
    """快捷获取项目根目录"""
    return get_config().get_path('paths.project_root')

def get_output_dir() -> Path:
    """快捷获取输出目录"""
    return get_config().get_path('paths.output_dir')


if __name__ == '__main__':
    # 测试代码
    config = ProjectConfig()
    
    print("=== 配置加载器测试 ===\n")
    
    print("1. 路径配置:")
    print(f"   项目根目录: {config.get_path('paths.project_root')}")
    print(f"   数据目录: {config.get_path('paths.data_dir')}")
    print(f"   输出目录: {config.get_path('paths.output_dir')}")
    
    print("\n2. 基因配置:")
    print(f"   Tier1基因: {config.get_genes('tier1')}")
    print(f"   所有候选基因: {config.get_all_candidate_genes()}")
    
    print("\n3. 参数配置:")
    print(f"   World Model hidden_dim: {config.get_parameter('world_model.hidden_dim')}")
    print(f"   Hub阈值百分位: {config.get_parameter('hub_detection.hub_threshold_percentile')}")
    
    print("\n4. 基因注释:")
    print(f"   TOR1AIP1功能: {config.get_gene_annotation('TOR1AIP1', 'function')}")
    print(f"   TOR1AIP1临床意义: {config.get_gene_annotation('TOR1AIP1', 'clinical_significance')}")
    
    print("\n5. 基因-生物标志物映射:")
    print(f"   TOR1AIP1对应生物标志物: {config.get_gene_biomarkers('TOR1AIP1')}")
    
    print("\n✅ 配置加载器测试通过！")
