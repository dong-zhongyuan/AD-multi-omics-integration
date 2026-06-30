import os
#!/usr/bin/env python3
"""
统一的基因ID转换工具
- 支持Ensembl ID ↔ Gene Symbol互转
- 自动缓存到单一文件
- 支持批量查询和单个查询
- 优先使用缓存，缓存不存在时在线查询
"""

import pandas as pd
import requests
import time
from pathlib import Path
from typing import Dict, List, Union, Optional

# 全局缓存文件路径
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[2])
CACHE_FILE = PROJECT_ROOT / "data" / "metadata" / "gene_id_mapping_cache.csv"


class GeneIDConverter:
    """基因ID转换器（单例模式）"""
    
    _instance = None
    _cache = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化转换器，加载缓存"""
        if self._cache is None:
            self._load_cache()
    
    def _load_cache(self):
        """加载缓存文件"""
        if CACHE_FILE.exists():
            df = pd.read_csv(CACHE_FILE)
            # 创建双向映射
            self._cache = {
                'ensembl_to_symbol': dict(zip(df['ensembl_id'], df['gene_symbol'])),
                'symbol_to_ensembl': dict(zip(df['gene_symbol'], df['ensembl_id']))
            }
            print(f"✅ 加载基因ID缓存: {len(df)} 个映射")
        else:
            self._cache = {
                'ensembl_to_symbol': {},
                'symbol_to_ensembl': {}
            }
            print("⚠️ 缓存文件不存在，将在线查询")
    
    def _save_cache(self):
        """保存缓存到文件"""
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # 合并双向映射
        all_mappings = {}
        for ens_id, symbol in self._cache['ensembl_to_symbol'].items():
            all_mappings[ens_id] = symbol
        
        df = pd.DataFrame({
            'ensembl_id': list(all_mappings.keys()),
            'gene_symbol': list(all_mappings.values())
        }).sort_values('gene_symbol')
        
        df.to_csv(CACHE_FILE, index=False)
        print(f"💾 保存基因ID缓存: {len(df)} 个映射 → {CACHE_FILE}")
    
    def _query_ensembl_api(self, ensembl_id: str) -> Optional[str]:
        """在线查询Ensembl API"""
        try:
            url = f"https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                symbol = data.get('display_name')
                if symbol:
                    return symbol
            
            time.sleep(0.1)  # 避免API限流
        except Exception as e:
            print(f"  ⚠️ 查询失败 {ensembl_id}: {e}")
        
        return None
    
    def _query_mygene(self, ensembl_ids: List[str]) -> Dict[str, str]:
        """使用mygene批量查询（更快）"""
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            
            results = mg.querymany(
                ensembl_ids,
                scopes='ensembl.gene',
                fields='symbol',
                species='human',
                returnall=True
            )
            
            mapping = {}
            for item in results['out']:
                if 'symbol' in item and 'query' in item:
                    mapping[item['query']] = item['symbol']
            
            return mapping
        except ImportError:
            print("  ⚠️ mygene未安装，使用Ensembl API")
            return {}
        except Exception as e:
            print(f"  ⚠️ mygene查询失败: {e}")
            return {}
    
    def ensembl_to_symbol(
        self,
        ensembl_ids: Union[str, List[str]],
        use_cache: bool = True,
        save_cache: bool = True
    ) -> Union[str, Dict[str, str]]:
        """
        Ensembl ID → Gene Symbol
        
        Args:
            ensembl_ids: 单个ID或ID列表
            use_cache: 是否使用缓存
            save_cache: 是否保存新查询到缓存
        
        Returns:
            单个ID返回str，多个ID返回dict
        """
        # 处理单个ID
        if isinstance(ensembl_ids, str):
            if use_cache and ensembl_ids in self._cache['ensembl_to_symbol']:
                return self._cache['ensembl_to_symbol'][ensembl_ids]
            
            # 在线查询
            symbol = self._query_ensembl_api(ensembl_ids)
            if symbol and save_cache:
                self._cache['ensembl_to_symbol'][ensembl_ids] = symbol
                self._cache['symbol_to_ensembl'][symbol] = ensembl_ids
                self._save_cache()
            
            return symbol or ensembl_ids
        
        # 处理ID列表
        mapping = {}
        missing_ids = []
        
        # 先从缓存获取
        if use_cache:
            for ens_id in ensembl_ids:
                if ens_id in self._cache['ensembl_to_symbol']:
                    mapping[ens_id] = self._cache['ensembl_to_symbol'][ens_id]
                else:
                    missing_ids.append(ens_id)
        else:
            missing_ids = list(ensembl_ids)
        
        # 批量查询缺失的ID
        if missing_ids:
            print(f"🔍 查询 {len(missing_ids)} 个新的Ensembl ID...")
            
            # 优先使用mygene（更快）
            new_mapping = self._query_mygene(missing_ids)
            
            # mygene失败的用Ensembl API逐个查询
            still_missing = [eid for eid in missing_ids if eid not in new_mapping]
            if still_missing:
                print(f"  使用Ensembl API查询剩余 {len(still_missing)} 个ID...")
                for ens_id in still_missing:
                    symbol = self._query_ensembl_api(ens_id)
                    if symbol:
                        new_mapping[ens_id] = symbol
                    else:
                        new_mapping[ens_id] = ens_id  # 使用原ID
            
            # 更新缓存
            if save_cache and new_mapping:
                self._cache['ensembl_to_symbol'].update(new_mapping)
                for ens_id, symbol in new_mapping.items():
                    self._cache['symbol_to_ensembl'][symbol] = ens_id
                self._save_cache()
            
            mapping.update(new_mapping)
        
        return mapping
    
    def symbol_to_ensembl(
        self,
        symbols: Union[str, List[str]],
        use_cache: bool = True
    ) -> Union[str, Dict[str, str]]:
        """
        Gene Symbol → Ensembl ID
        
        Args:
            symbols: 单个symbol或symbol列表
            use_cache: 是否使用缓存
        
        Returns:
            单个symbol返回str，多个symbol返回dict
        """
        # 处理单个symbol
        if isinstance(symbols, str):
            if use_cache and symbols in self._cache['symbol_to_ensembl']:
                return self._cache['symbol_to_ensembl'][symbols]
            return None
        
        # 处理symbol列表
        mapping = {}
        if use_cache:
            for symbol in symbols:
                if symbol in self._cache['symbol_to_ensembl']:
                    mapping[symbol] = self._cache['symbol_to_ensembl'][symbol]
        
        return mapping
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'total_mappings': len(self._cache['ensembl_to_symbol']),
            'cache_file': str(CACHE_FILE),
            'cache_exists': CACHE_FILE.exists()
        }


# 全局单例实例
_converter = None

def get_converter() -> GeneIDConverter:
    """获取全局转换器实例"""
    global _converter
    if _converter is None:
        _converter = GeneIDConverter()
    return _converter


# 便捷函数
def ensembl_to_symbol(ensembl_ids: Union[str, List[str]], **kwargs) -> Union[str, Dict[str, str]]:
    """Ensembl ID → Gene Symbol（便捷函数）"""
    return get_converter().ensembl_to_symbol(ensembl_ids, **kwargs)


def symbol_to_ensembl(symbols: Union[str, List[str]], **kwargs) -> Union[str, Dict[str, str]]:
    """Gene Symbol → Ensembl ID（便捷函数）"""
    return get_converter().symbol_to_ensembl(symbols, **kwargs)


if __name__ == "__main__":
    # 测试
    print("=" * 80)
    print("基因ID转换工具测试")
    print("=" * 80)
    
    converter = get_converter()
    
    # 测试单个ID
    print("\n1. 测试单个ID转换:")
    test_id = "ENSG00000186868"  # MAPT
    symbol = converter.ensembl_to_symbol(test_id)
    print(f"  {test_id} → {symbol}")
    
    # 测试批量转换
    print("\n2. 测试批量转换:")
    test_ids = ["ENSG00000186868", "ENSG00000186318", "ENSG00000080815"]  # MAPT, BACE1, PSEN1
    mapping = converter.ensembl_to_symbol(test_ids)
    for ens_id, sym in mapping.items():
        print(f"  {ens_id} → {sym}")
    
    # 显示缓存统计
    print("\n3. 缓存统计:")
    stats = converter.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
