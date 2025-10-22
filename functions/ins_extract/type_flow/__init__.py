"""
Type flowmodule - Type Flow module
Different insulator extraction strategies for different tower types
"""
from .ins_extract_zl import ins_extract_zl
from .ins_extract_zl1 import ins_extract_zl1
from .ins_extract_type4 import ins_extract_type4
from .ins_extract_type51 import ins_extract_type51

__all__ = ['ins_extract_zl', 'ins_extract_zl1', 'ins_extract_type4', 'ins_extract_type51']
