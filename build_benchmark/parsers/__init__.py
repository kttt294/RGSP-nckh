"""Package chứa các bộ đọc nhãn chuẩn hóa cho 5 dataset của RS-Solve."""
from .parse_dota import parse_dota_label, parse_dota_header, DOTA_CLASS_MAP
from .parse_isaid import parse_isaid_label, ISAID_NAME_MAP
from .parse_dior import parse_dior_label, DIOR_NAME_MAP
from .parse_xview import parse_xview_label, XVIEW_ID_MAP
from .parse_visdrone import parse_visdrone_label, VISDRONE_NAME_MAP

__all__ = [
    "parse_dota_label",
    "parse_dota_header",
    "parse_isaid_label",
    "parse_dior_label",
    "parse_xview_label",
    "parse_visdrone_label",
    "DOTA_CLASS_MAP",
    "ISAID_NAME_MAP",
    "DIOR_NAME_MAP",
    "XVIEW_ID_MAP",
    "VISDRONE_NAME_MAP",
]
