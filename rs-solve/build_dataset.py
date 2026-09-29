"""Bộ điều khiển chính (CLI Runner) cho tập dữ liệu RS-Solve đa nguồn.

Hỗ trợ sinh câu hỏi trắc nghiệm kiểm soát vật lý cho cả 5 dataset:
- DOTA (v1.0, v1.5, v2.0)
- iSAID (COCO Instance Segmentation)
- DIOR (Pascal VOC XML)
- xView (GeoJSON)
- VisDrone (TXT)

Cách chạy:
    python build_dataset.py --dataset dota --per-kind 1
    python build_dataset.py --dataset all --per-kind 5 --output-prefix mau_35_all
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from PIL import Image

from core_generators import (
    Scene, InvalidInput, SkipSample, GENERATORS, generate_q6,
    ABSTAIN, KINDS, DEFAULT_CALIBRATION, DEFAULT_DOTA_CLASSES
)
from parsers import (
    parse_dota_label, parse_dota_header,
    parse_isaid_label, parse_dior_label,
    parse_xview_label, parse_visdrone_label,
    DOTA_CLASS_MAP, ISAID_NAME_MAP, DIOR_NAME_MAP,
    XVIEW_ID_MAP, VISDRONE_NAME_MAP
)

BASE_DIR = Path(__file__).resolve().parent

# GSD mặc định hoặc tham chiếu cho từng dataset
DATASET_DEFAULT_GSD = {
    "dota": 0.3,
    "isaid": 0.3,
    "dior": 1.0,
    "xview": 0.3,       # Cố định WorldView-3
    "visdrone": 0.1,    # Ước lượng từ độ cao bay flycam
}

DATASET_CANONICAL_CLASSES = {
    "dota": list(DOTA_CLASS_MAP.values()),
    "visdrone": list(VISDRONE_NAME_MAP.values()),
    "dior": list(DIOR_NAME_MAP.values()),
    "isaid": list(ISAID_NAME_MAP.values()),
    "xview": list(set(XVIEW_ID_MAP.values())),
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_specs(path=BASE_DIR / "physical_sizes.json"):
    data = load_json(path)
    if isinstance(data, dict):
        return data
    return {row["class_id"]: row for row in data}


def load_config(path=BASE_DIR / "build_config.json"):
    cfg = load_json(path)
    if "calibration" not in cfg:
        cfg["calibration"] = {k: {"p0_U_px": v} for k, v in DEFAULT_CALIBRATION.items()}
    return cfg


def get_scene_meta(config, img_path):
    """Tìm siêu dữ liệu cho ảnh từ cấu hình images hoặc scenes."""
    store = config.get("images") or config.get("scenes") or {}
    name = Path(img_path).name
    if str(img_path) in store:
        return dict(store[str(img_path)])
    for k, v in store.items():
        if Path(k).name == name:
            return dict(v)
    return {}


def to_relative_posix(path, root=None):
    """Chuyển đổi đường dẫn thành đường dẫn tương đối với định dạng POSIX (dấu /)."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()

    if root is not None:
        try:
            return p.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass

    try:
        return p.resolve().relative_to(BASE_DIR.resolve()).as_posix()
    except ValueError:
        pass

    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        pass

    return p.as_posix()


def load_dataset_scene(dataset_name, img_path, label_path, meta, config, root_dir=None):
    """Nạp một ảnh và tệp nhãn tương ứng theo parser chuyên biệt của từng dataset."""
    img_p = Path(img_path)
    lbl_p = Path(label_path)
    if not img_p.is_file():
        raise InvalidInput(f"Ảnh không tồn tại: {img_path}")
    if not lbl_p.is_file():
        raise InvalidInput(f"Nhãn không tồn tại: {label_path}")

    with Image.open(img_p) as img:
        img_w, img_h = img.size
        pixels = img.copy()

    # Xác định GSD
    gsd_source = "dataset_default"
    gsd_m = DATASET_DEFAULT_GSD.get(dataset_name, 0.3)
    
    if dataset_name in ("dota", "isaid"):
        header = parse_dota_header(lbl_p)
        if "gsd" in header:
            try:
                gsd_m = float(header["gsd"])
                gsd_source = "label_header"
            except ValueError:
                pass
    elif dataset_name == "xview":
        gsd_m = 0.3
        gsd_source = "worldview3_sensor_fixed"
    elif dataset_name == "visdrone":
        gsd_source = "uav_flight_altitude_estimated"

    # Ưu tiên cao nhất: nếu người dùng cấu hình rõ gsd_m trong metadata
    if "gsd_m" in meta and meta["gsd_m"] is not None:
        try:
            gsd_m = float(meta["gsd_m"])
            gsd_source = meta.get("gsd_source", "metadata_override")
        except (ValueError, TypeError):
            pass

    # Gọi parser chuyên biệt
    ds = dataset_name.lower()
    if ds == "dota":
        fmt = meta.get("label_format", "auto")
        objects = parse_dota_label(lbl_p, img_w, img_h, label_format=fmt)
    elif ds == "isaid":
        objects = parse_isaid_label(lbl_p, img_p.name, img_w, img_h)
    elif ds == "dior":
        objects = parse_dior_label(lbl_p, img_w, img_h)
    elif ds == "xview":
        objects = parse_xview_label(lbl_p, img_p.name, img_w, img_h)
    elif ds == "visdrone":
        objects = parse_visdrone_label(lbl_p, img_w, img_h)
    else:
        raise InvalidInput(f"Dataset chưa được hỗ trợ parser: {dataset_name}")

    scale = float(meta.get("scale", 1.0))
    if "source_classes" not in meta:
        meta["source_classes"] = DATASET_CANONICAL_CLASSES.get(ds, list({o["class_id"] for o in objects}))
    if "complete_classes" not in meta:
        meta["complete_classes"] = list(meta["source_classes"])
    if "annotation_review_source" not in meta:
        meta["annotation_review_source"] = f"official {dataset_name} benchmark exhaustive annotation"

    rel_image_path = to_relative_posix(img_p, root=root_dir)

    return Scene(
        image_path=rel_image_path,
        img_w=img_w,
        img_h=img_h,
        gsd_m=gsd_m,
        scale=scale,
        meta=meta,
        objects=objects,
        pixels=pixels,
        gsd_source=gsd_source
    )


def discover_scenes(root_dir, dataset_filter="all"):
    """Tự động phát hiện các cặp (ảnh, nhãn, dataset) trong thư mục gốc, sample_data/ hoặc Kaggle input."""
    root = Path(root_dir)
    discovered = []
    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.JPG", "*.PNG")

    def find_images(folder):
        imgs = []
        for ext in image_exts:
            imgs.extend(folder.glob(ext))
        return sorted(imgs)
    
    # 1. Kiểm tra sample_data/ trước
    sample_dir = root / "sample_data"
    if sample_dir.is_dir():
        datasets = [dataset_filter] if dataset_filter != "all" else ["dota", "visdrone", "dior", "xview", "isaid"]
        for ds in datasets:
            ds_path = sample_dir / ds
            if not ds_path.is_dir():
                continue
            if ds == "dota":
                for img in find_images(ds_path / "images"):
                    lbl = ds_path / "labels" / f"{img.stem}.txt"
                    if lbl.exists():
                        discovered.append(("dota", img, lbl))
            elif ds == "visdrone":
                for img in find_images(ds_path / "images"):
                    lbl = ds_path / "annotations" / f"{img.stem}.txt"
                    if lbl.exists():
                        discovered.append(("visdrone", img, lbl))
            elif ds == "dior":
                img_dir = ds_path / "images" if (ds_path / "images").is_dir() else ds_path / "JPEGImages"
                lbl_dir = ds_path / "Annotations" if (ds_path / "Annotations").is_dir() else (
                    ds_path / "annotations" if (ds_path / "annotations").is_dir() else ds_path / "labels"
                )
                if img_dir.is_dir() and lbl_dir.is_dir():
                    for img in find_images(img_dir):
                        lbl = lbl_dir / f"{img.stem}.xml"
                        if lbl.exists():
                            discovered.append(("dior", img, lbl))
            elif ds == "xview":
                lbl = ds_path / "labels" / "xview_sample.geojson"
                for img in find_images(ds_path / "images"):
                    if lbl.exists():
                        discovered.append(("xview", img, lbl))
            elif ds == "isaid":
                lbl = ds_path / "annotations" / "iSAID_sample.json"
                for img in find_images(ds_path / "images"):
                    if lbl.exists():
                        discovered.append(("isaid", img, lbl))
                        
    # 2. Nếu không tìm thấy trong sample_data, tìm trong images/ và labels/ truyền thống
    if not discovered:
        img_dir = root / "images"
        lbl_dir = root / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            for img in find_images(img_dir):
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    discovered.append(("dota", img, lbl))

    # 3. Quét đệ quy thích ứng các dataset gắn trên Kaggle (/kaggle/input)
    if not discovered and root.is_dir():
        for sub in [p for p in root.iterdir() if p.is_dir()]:
            sub_name = sub.name.lower()
            # DOTA: tìm nhãn trong labelTxt hoặc labels
            for lbl_cand in list(sub.rglob("labelTxt")) + list(sub.rglob("labels")):
                if lbl_cand.is_dir():
                    parent = lbl_cand.parent
                    img_cand = parent / "images" if (parent / "images").is_dir() else sub / "images"
                    if img_cand.is_dir():
                        for img in find_images(img_cand):
                            txt = lbl_cand / f"{img.stem}.txt"
                            if txt.is_file():
                                discovered.append(("dota", img, txt))
            # DIOR: tìm Annotations và images / JPEGImages
            for ann_cand in list(sub.rglob("Annotations")) + list(sub.rglob("annotations")):
                if ann_cand.is_dir() and ("dior" in sub_name or "dior" in ann_cand.as_posix().lower()):
                    parent = ann_cand.parent
                    img_cand = parent / "images" if (parent / "images").is_dir() else (parent / "JPEGImages")
                    if not img_cand.is_dir() and (sub / "images").is_dir():
                        img_cand = sub / "images"
                    if img_cand.is_dir():
                        for img in find_images(img_cand):
                            xml = ann_cand / f"{img.stem}.xml"
                            if xml.is_file():
                                discovered.append(("dior", img, xml))
            # VisDrone: tìm annotations và images
            for ann_cand in sub.rglob("annotations"):
                if ann_cand.is_dir() and "visdrone" in sub_name:
                    parent = ann_cand.parent
                    img_cand = parent / "images" if (parent / "images").is_dir() else (sub / "images")
                    if img_cand.is_dir():
                        for img in find_images(img_cand):
                            txt = ann_cand / f"{img.stem}.txt"
                            if txt.is_file():
                                discovered.append(("visdrone", img, txt))
            # iSAID: tìm file COCO json và images
            for json_file in sub.rglob("*.json"):
                if "isaid" in sub_name or "isaid" in json_file.name.lower():
                    img_cand = json_file.parent / "images" if (json_file.parent / "images").is_dir() else (sub / "images")
                    if img_cand.is_dir():
                        for img in find_images(img_cand):
                            discovered.append(("isaid", img, json_file))
            # xView: tìm file geojson và images
            for geojson_file in sub.rglob("*.geojson"):
                img_cand = geojson_file.parent / "images" if (geojson_file.parent / "images").is_dir() else (sub / "images")
                if img_cand.is_dir():
                    for img in find_images(img_cand):
                        discovered.append(("xview", img, geojson_file))
                    
    if dataset_filter != "all":
        discovered = [item for item in discovered if item[0] == dataset_filter]

    return discovered


def build_dataset_multi(root_dir, config, specs, dataset_name="all", per_kind=1):
    """Quy trình sinh câu hỏi trắc nghiệm hoàn chỉnh từ các dataset được chọn."""
    discovered = discover_scenes(root_dir, dataset_name)
    if not discovered:
        raise InvalidInput(f"Không tìm thấy ảnh/nhãn nào cho dataset: {dataset_name} tại {root_dir}")

    scenes = []
    provenance = {}
    for ds, img_p, lbl_p in discovered:
        scene_meta = get_scene_meta(config, img_p)
        sc = load_dataset_scene(ds, img_p, lbl_p, scene_meta, config, root_dir=root_dir)
        scenes.append(sc)
        provenance[sc.image_path] = {
            "dataset": ds,
            "gsd_m": sc.gsd_m,
            "gsd_source": sc.gsd_source,
            "objects_count": len(sc.objects),
            "classes": sorted(list({o["class_id"] for o in sc.objects}))
        }

    pool = {k: [] for k in KINDS}
    rejects = []

    # Sinh Q1 đến Q5
    for sc in scenes:
        for generator in GENERATORS:
            for sample in generator(sc, specs, config, rejects):
                pool[sample["kind"]].append(sample)

    # Sinh Q6 từ pool Q3-HF, Q4, Q5
    for base in pool["Q3-HF"] + pool["Q4"] + pool["Q5"]:
        q6 = generate_q6(base)
        if q6:
            pool["Q6"].append(q6)

    # Cắt quota theo per_kind
    selected = []
    kind_counts = {}
    for kind in KINDS:
        items = pool[kind][:per_kind]
        kind_counts[kind] = len(items)
        selected.extend(items)

    # Đánh số ID
    counters = Counter()
    for item in selected:
        counters[item["kind"]] += 1
        item["id"] = f"RS-SOLVE-{item['kind'].replace('-', '')}-{counters[item['kind']]:03d}"

    report = {
        "dataset_filter": dataset_name,
        "complete": all(kind_counts[k] >= per_kind for k in KINDS),
        "requested_per_kind": per_kind,
        "counts_by_kind": kind_counts,
        "total_questions": len(selected),
        "scenes_processed": len(scenes),
        "scene_provenance": provenance,
        "rejects_count": len(rejects),
        "rejects_sample": rejects[:20],
    }

    return selected, report


def main():
    parser = argparse.ArgumentParser(description="Sinh benchmark RS-Solve từ đa dataset.")
    parser.add_argument("--dataset", default="all", choices=["all", "dota", "isaid", "dior", "xview", "visdrone"],
                        help="Tên dataset cần trích xuất (mặc định: all)")
    parser.add_argument("--per-kind", type=int, default=1, help="Số câu hỏi tối thiểu cho mỗi kind (mặc định: 1)")
    parser.add_argument("--output-prefix", default="rs_solve_questions", help="Tiền tố file xuất ra (json/jsonl)")
    parser.add_argument("--data-dir", default=str(BASE_DIR), help="Thư mục dữ liệu gốc")
    args = parser.parse_args()

    config = load_config()
    specs = load_specs()

    print(f"=== BẮT ĐẦU SINH DỮ LIỆU RS-SOLVE [{args.dataset.upper()}] ===")
    samples, report = build_dataset_multi(args.data_dir, config, specs, args.dataset, args.per_kind)

    out_json = BASE_DIR / f"{args.output_prefix}.json"
    out_jsonl = BASE_DIR / f"{args.output_prefix}.jsonl"
    out_rep = BASE_DIR / f"{args.output_prefix}_build_report.json"

    out_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    out_rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Hoàn thành: {len(samples)} câu hỏi được tạo.")
    print(f"Thống kê theo kind: {report['counts_by_kind']}")
    print(f"Đã lưu: {out_json.name}, {out_jsonl.name}, {out_rep.name}")


if __name__ == "__main__":
    main()
