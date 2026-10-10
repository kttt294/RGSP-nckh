"""Tool tải ảnh và phục vụ kiểm tra thủ công nhãn Q3-LF (RS-Solve Benchmark).

Chức năng:
1. Tải chính xác các ảnh được chọn cho Q3-LF từ Kaggle (không cần tải toàn bộ dataset hàng chục GB).
2. Tự động cắt (crop) đối tượng mục tiêu ra thư mục `crops/` để kiểm tra màu trực quan trong 1 giây.
3. Tự động vẽ bounding box và lưới 3x3 lên ảnh toàn cảnh `annotated/`.
4. Xuất báo cáo web tương tác `inspect.html` để kiểm tra thủ công thuận tiện trên trình duyệt.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Vui lòng cài đặt kaggle: pip install kaggle")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Tải ảnh và sinh công cụ kiểm tra nhãn Q3-LF.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot",
                        help="Tải ảnh cho tập Pilot (143 câu, 130 ảnh) hoặc Full (439 câu, 339 ảnh). Mặc định: pilot")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Giới hạn số ảnh cần tải (hữu ích khi muốn chạy thử trước vài ảnh)")
    parser.add_argument("--skip_download", action="store_true",
                        help="Bỏ qua bước tải, chỉ chạy tạo crop và giao diện HTML nếu đã có ảnh trong raw_images/")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    repo_dir = base_dir.parent if base_dir.name in ("scripts", "data") else base_dir
    work_dir = repo_dir / "data" / "q3_lf_inspection"

    raw_dir = work_dir / "raw_images"
    crop_dir = work_dir / "crops"
    annotated_dir = work_dir / "annotated"
    temp_dir = work_dir / "temp_dl"

    raw_dir.mkdir(exist_ok=True, parents=True)
    crop_dir.mkdir(exist_ok=True, parents=True)
    annotated_dir.mkdir(exist_ok=True, parents=True)
    temp_dir.mkdir(exist_ok=True, parents=True)

    csv_file = repo_dir / "data" / f"q3_lf_{args.mode}_verification_list.csv"
    if not csv_file.exists():
        print(f"Không tìm thấy file danh sách: {csv_file}")
        sys.exit(1)

    with open(csv_file, "r", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))

    print(f"=== BẮT ĐẦU KIỂM TRA NHÃN Q3-LF (Chế độ: {args.mode.upper()}) ===")
    print(f"Tổng số câu hỏi: {len(records)}")

    # Nhóm theo ảnh độc nhất
    unique_images = {}
    for r in records:
        rel_path = r["relative_file_path"]
        if rel_path not in unique_images:
            unique_images[rel_path] = {
                "dataset": r["kaggle_dataset"],
                "filename": r["image_filename"],
                "questions": []
            }
        unique_images[rel_path]["questions"].append(r)

    print(f"Tổng số ảnh độc nhất cần tải: {len(unique_images)}")

    # Khởi tạo Kaggle API
    api = None
    if not args.skip_download:
        print("Đang xác thực Kaggle API...")
        try:
            api = KaggleApi()
            api.authenticate()
            print("Xác thực Kaggle API thành công!")
        except Exception as e:
            print(f"Lỗi xác thực Kaggle: {e}")
            sys.exit(1)

    items_to_process = list(unique_images.items())
    if args.max_images:
        items_to_process = items_to_process[:args.max_images]
        print(f"Đã giới hạn tải {args.max_images} ảnh theo tham số --max_images.")

    # 1. Tải ảnh
    downloaded_count = 0
    for idx, (rel_path, info) in enumerate(items_to_process, 1):
        filename = info["filename"]
        target_raw = raw_dir / filename

        if target_raw.exists() and target_raw.stat().st_size > 0:
            # Đã tải trước đó
            continue

        if args.skip_download:
            continue

        ds_slug = info["dataset"]
        print(f"[{idx}/{len(items_to_process)}] Đang tải: {filename} từ {ds_slug}...")
        try:
            api.dataset_download_file(ds_slug, rel_path, path=str(temp_dir), quiet=True)
            # Giải nén mọi file .zip trong temp_dir
            for zf_path in temp_dir.glob("*.zip"):
                try:
                    with zipfile.ZipFile(zf_path, "r") as zf:
                        zf.extractall(path=temp_dir)
                except Exception as ze:
                    print(f"  Lỗi giải nén {zf_path.name}: {ze}")
                zf_path.unlink(missing_ok=True)

            # Tìm file ảnh đã giải nén
            extracted_file = None
            direct_candidate = temp_dir / filename
            if direct_candidate.exists():
                extracted_file = direct_candidate
            else:
                for f in temp_dir.rglob(filename):
                    extracted_file = f
                    break

            if extracted_file and extracted_file.exists():
                shutil.move(str(extracted_file), str(target_raw))
                downloaded_count += 1
            else:
                print(f"  [CẢNH BÁO] Không tìm thấy file {filename} sau giải nén.")
        except Exception as e:
            print(f"  [LỖI] Tải thất bại {filename}: {e}")
        finally:
            # Dọn dẹp temp
            for f in temp_dir.glob("*"):
                if f.is_file(): f.unlink(missing_ok=True)

    print(f"Hoàn thành tải: {downloaded_count} ảnh mới. Đã có sẵn: {len(items_to_process) - downloaded_count} ảnh.")

    # 2. Tạo Crop và Annotation
    print("\nĐang tạo các bản crop đối tượng và ảnh trực quan...")
    processed_q_count = 0
    inspection_items = []

    for rel_path, info in unique_images.items():
        filename = info["filename"]
        raw_img_path = raw_dir / filename

        if not raw_img_path.exists():
            continue

        try:
            img = Image.open(raw_img_path).convert("RGB")
            img_w, img_h = img.size
        except Exception as e:
            print(f"Lỗi đọc ảnh {filename}: {e}")
            continue

        # Vẽ lưới 3x3 và bbox cho từng câu hỏi
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)

        # Lưới 3x3
        for i in range(1, 3):
            x = (img_w / 3.0) * i
            y = (img_h / 3.0) * i
            draw.line([(x, 0), (x, img_h)], fill=(255, 255, 0, 128), width=2)
            draw.line([(0, y), (img_w, y)], fill=(255, 255, 0, 128), width=2)

        for q in info["questions"]:
            qid = q["id"]
            box_str = q.get("target_box_xyxy", "[]")
            try:
                box = eval(box_str)
            except Exception:
                box = []

            crop_rel_name = ""
            if len(box) == 4:
                x1, y1, x2, y2 = [float(v) for v in box]
                # Bounding box trên ảnh annotated
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0, y1 - 15)), f"{q['class_id']}: {q['ground_truth_color']}", fill="red")

                # Cắt crop có thêm lề 20%
                bw = x2 - x1
                bh = y2 - y1
                pad_x = max(8.0, bw * 0.2)
                pad_y = max(8.0, bh * 0.2)
                cx1 = max(0, int(x1 - pad_x))
                cy1 = max(0, int(y1 - pad_y))
                cx2 = min(img_w, int(x2 + pad_x))
                cy2 = min(img_h, int(y2 + pad_y))

                crop_img = img.crop((cx1, cy1, cx2, cy2))
                crop_filename = f"{qid}_{q['class_id']}_{q['ground_truth_color'].replace(' ', '_').replace('/', '_')}.png"
                crop_path = crop_dir / crop_filename
                crop_img.save(crop_path)
                crop_rel_name = crop_filename

            annotated_filename = f"annotated_{Path(filename).stem}.jpg"
            annotated_path = annotated_dir / annotated_filename
            annotated_img.convert("RGB").save(annotated_path, "JPEG", quality=85)

            inspection_items.append({
                "id": qid,
                "dataset": q["dataset"],
                "class_id": q["class_id"],
                "target_cell": q["target_cell"],
                "ground_truth_color": q["ground_truth_color"],
                "rho_px": q["rho_px"],
                "crop_img": crop_rel_name,
                "annotated_img": annotated_filename,
                "question": q["question"]
            })
            processed_q_count += 1

    # 3. Tạo trang web kiểm tra HTML tương tác
    html_file_work = work_dir / "inspect.html"
    html_file_scripts = repo_dir / "scripts" / "inspect.html"
    create_html_inspection_page(html_file_work, inspection_items, args.mode)
    create_html_inspection_page(html_file_scripts, inspection_items, args.mode)

    # Dọn dẹp temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n HOÀN THÀNH!")
    print(f"- Số ảnh gốc tải về: {len(list(raw_dir.glob('*')))} file tại: {raw_dir}")
    print(f"- Số mẫu crop đối tượng: {len(list(crop_dir.glob('*')))} file tại: {crop_dir}")
    print(f"- Giao diện kiểm tra tương tác đã lưu tại:")
    print(f"  + file:///{html_file_scripts.resolve()}")
    print(f"  + file:///{html_file_work.resolve()}")


def create_html_inspection_page(html_path: Path, items: list[dict], mode: str):
    """Sinh file HTML cho phép người dùng click duyệt và kiểm tra Đúng/Sai cho từng nhãn màu."""
    items_json = json.dumps(items, ensure_ascii=False)
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>RS-Solve: Kiểm tra nhãn Q3-LF ({mode.upper()})</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f8f9fa; color: #212529; }}
  h1 {{ font-size: 22px; color: #1e3a8a; margin-bottom: 5px; }}
  .summary {{ background: #fff; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .toolbar {{ margin-bottom: 15px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  input[type="text"], select {{ padding: 6px 12px; border: 1px solid #ced4da; border-radius: 4px; font-size: 14px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e9ecef; font-size: 13px; }}
  th {{ background: #2563eb; color: #fff; font-weight: 600; position: sticky; top: 0; }}
  tr:hover {{ background: #f1f5f9; }}
  .crop-preview {{ width: 80px; height: 80px; object-fit: contain; background: #222; border-radius: 4px; border: 1px solid #ccc; cursor: pointer; }}
  .color-badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-weight: bold; font-size: 12px; }}
  .color-Màu_đỏ {{ background: #fee2e2; color: #991b1b; }}
  .color-Màu_vàng {{ background: #fef3c7; color: #92400e; }}
  .color-Màu_xanh_dương {{ background: #dbeafe; color: #1e40af; }}
  .color-Màu_xanh_lá_cây {{ background: #dcfce7; color: #166534; }}
  .color-Màu_trắng {{ background: #f3f4f6; color: #374151; border: 1px solid #d1d5db; }}
  .color-Màu_đen___xám {{ background: #374151; color: #f9fafb; }}
  .color-Không_thể_xác_định_từ_ảnh {{ background: #f3e8ff; color: #6b21a8; }}
  .btn-export {{ background: #10b981; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer; }}
  .btn-export:hover {{ background: #059669; }}
  .status-btn {{ padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; cursor: pointer; font-size: 12px; }}
  .status-correct {{ background: #dcfce7; color: #166534; border-color: #86efac; }}
  .status-wrong {{ background: #fee2e2; color: #991b1b; border-color: #fca5a5; }}
  /* Modal zoom */
  .modal {{ display: none; position: fixed; z-index: 999; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); justify-content: center; align-items: center; }}
  .modal img {{ max-width: 90%; max-height: 90%; border-radius: 8px; }}
</style>
</head>
<body>

<h1>🔍 RS-Solve: Công Cụ Kiểm Tra Thủ Công Nhãn Q3-LF ({mode.upper()})</h1>
<div class="summary">
  <b>Hướng dẫn:</b> Xem hình cắt (crop) đối tượng ở cột <b>Ảnh đối tượng</b> để kiểm tra màu sắc thực tế so với <b>Nhãn gán tự động</b>.
  Nhấp vào ảnh để phóng to. Đánh dấu <b>Đúng / Sai</b> ở cột Kiểm tra để lưu lại đánh giá của bạn.
</div>

<div class="toolbar">
  <input type="text" id="searchInput" placeholder="Tìm kiếm theo ID, Class, Màu..." oninput="filterTable()">
  <select id="colorFilter" onchange="filterTable()">
    <option value="">-- Tất cả màu --</option>
    <option value="Màu đỏ">Màu đỏ</option>
    <option value="Màu vàng">Màu vàng</option>
    <option value="Màu xanh dương">Màu xanh dương</option>
    <option value="Màu xanh lá cây">Màu xanh lá cây</option>
    <option value="Màu trắng">Màu trắng</option>
    <option value="Màu đen / xám">Màu đen / xám</option>
    <option value="Không thể xác định từ ảnh">Không thể xác định</option>
  </select>
  <button class="btn-export" onclick="exportAuditResult()">Xuất kết quả kiểm tra (JSON)</button>
  <span id="counter" style="margin-left: auto; font-size: 14px; font-weight: bold;"></span>
</div>

<table id="auditTable">
  <thead>
    <tr>
      <th>STT</th>
      <th>Mã câu hỏi</th>
      <th>Dataset</th>
      <th>Ảnh đối tượng (Crop)</th>
      <th>Lớp đối tượng</th>
      <th>Ô lưới</th>
      <th>Độ phân giải (ρ_px)</th>
      <th>Nhãn gán tự động</th>
      <th>Kiểm tra của bạn</th>
      <th>Ghi chú</th>
    </tr>
  </thead>
  <tbody id="tableBody"></tbody>
</table>

<div id="imageModal" class="modal" onclick="closeModal()">
  <img id="modalImg" src="">
</div>

<script>
  const data = {items_json};
  let auditStatus = {{}};

  function resolveAsset(subfolder, filename) {{
    if (!filename) return '';
    const p = window.location.pathname.replace(/\\\\/g, '/');
    const inScripts = p.includes('/scripts/');
    const prefix = inScripts ? '../data/q3_lf_inspection/' : '';
    return prefix + subfolder + '/' + encodeURI(filename);
  }}

  function handleImgError(img, subfolder, filename) {{
    if (!img.dataset.retried) {{
      img.dataset.retried = '1';
      if (img.src.includes('data/q3_lf_inspection')) {{
        img.src = subfolder + '/' + encodeURI(filename);
      }} else {{
        img.src = '../data/q3_lf_inspection/' + subfolder + '/' + encodeURI(filename);
      }}
    }}
  }}

  function renderTable() {{
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    const search = document.getElementById('searchInput').value.toLowerCase();
    const colorFilter = document.getElementById('colorFilter').value;

    let shownCount = 0;
    data.forEach((item, idx) => {{
      const matchSearch = item.id.toLowerCase().includes(search) || 
                          item.class_id.toLowerCase().includes(search) ||
                          item.ground_truth_color.toLowerCase().includes(search);
      const matchColor = !colorFilter || item.ground_truth_color === colorFilter;

      if (!matchSearch || !matchColor) return;
      shownCount++;

      const tr = document.createElement('tr');
      const colorClass = 'color-' + item.ground_truth_color.replace(/\\s+/g, '_').replace(/\\//g, '_');
      const cropSrc = item.crop_img ? resolveAsset('crops', item.crop_img) : '';
      const fullSrc = item.annotated_img ? resolveAsset('annotated', item.annotated_img) : '';

      const currentStatus = auditStatus[item.id] || 'pending';

      tr.innerHTML = `
        <td>${{idx + 1}}</td>
        <td><b>${{item.id}}</b></td>
        <td>${{item.dataset}}</td>
        <td>
          ${{cropSrc ? `<img src="${{cropSrc}}" class="crop-preview" onerror="handleImgError(this, 'crops', '${{item.crop_img}}')" onclick="openModal('${{fullSrc || cropSrc}}')" title="Bấm để xem ảnh toàn cảnh">` : '<i>Chưa có ảnh</i>'}}
        </td>
        <td><code>${{item.class_id}}</code></td>
        <td><b>${{item.target_cell}}</b></td>
        <td>${{parseFloat(item.rho_px).toFixed(1)}} px</td>
        <td><span class="color-badge ${{colorClass}}">${{item.ground_truth_color}}</span></td>
        <td>
          <button class="status-btn ${{currentStatus === 'correct' ? 'status-correct' : ''}}" onclick="setStatus('${{item.id}}', 'correct')">✓ Đúng</button>
          <button class="status-btn ${{currentStatus === 'wrong' ? 'status-wrong' : ''}}" onclick="setStatus('${{item.id}}', 'wrong')">✗ Sai</button>
        </td>
        <td>
          <input type="text" style="width: 140px;" placeholder="Ghi chú..." onchange="setNote('${{item.id}}', this.value)">
        </td>
      `;
      tbody.appendChild(tr);
    }});

    document.getElementById('counter').innerText = `Hiển thị: ${{shownCount}} / ${{data.length}} mẫu`;
  }}

  function setStatus(id, st) {{
    auditStatus[id] = st;
    renderTable();
  }}

  let auditNotes = {{}};
  function setNote(id, note) {{
    auditNotes[id] = note;
  }}

  function filterTable() {{
    renderTable();
  }}

  function openModal(src) {{
    document.getElementById('modalImg').src = src;
    document.getElementById('imageModal').style.display = 'flex';
  }}

  function closeModal() {{
    document.getElementById('imageModal').style.display = 'none';
  }}

  function exportAuditResult() {{
    const result = data.map(item => ({{
      id: item.id,
      ground_truth_color: item.ground_truth_color,
      user_verified_status: auditStatus[item.id] || 'unreviewed',
      user_notes: auditNotes[item.id] || ''
    }}));
    const blob = new Blob([JSON.stringify(result, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `q3_lf_audit_result_${{new Date().toISOString().slice(0,10)}}.json`;
    a.click();
  }}

  renderTable();
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    main()
