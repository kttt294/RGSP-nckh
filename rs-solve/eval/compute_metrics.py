"""Bộ tính toán chỉ số đánh giá khoa học (Metrics Computation Engine) theo chuẩn IEEE TGRS.

Tính toán toàn diện các Bảng VI, VII, VIII, IX, X theo kế hoạch nghiên cứu:
- BẢNG VI  : Ma trận kết cục (ρ_px hàng x ρ_tok cột) - % Hallucination / % Lỗi nhận thức / % Từ chối
- BẢNG VII : Mô hình hồi quy logistic kiểm định giả thuyết H1 (Hệ số β₁ log ρ_tok, p-value, 95% CI)
- BẢNG VIII: Hai ngưỡng phân giải H2 (Ngưỡng cảm biến p₀⁵⁰, p₀^U và ngưỡng token ρ*_tok, ρ* · P)
- BẢNG IX  : Bậc thang Johnson H3 & Kiểm định đơn điệu Jonckheere-Terpstra (Q3-LF < Q1 < Q4 < Q2 < Q5)
- BẢNG X   : Dự đoán có chọn lọc H4 (Risk-Coverage, Cov@5%, Cov@10%, Cov@20%, AURC, Abst.-acc trên U, Từ chối thừa trên A, Sel. Acc)
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def load_predictions(file_path: Path) -> List[Dict[str, Any]]:
    """Đọc dữ liệu dự đoán từ file JSONL."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except Exception as e:
                print(f"[!] Cảnh báo: Lỗi parse dòng {line_num} tại {file_path.name}: {e}")
    return records


# ==============================================================================
# 1. BẢNG VI: MA TRẬN KẾT CỤC ρ_px x ρ_tok
# ==============================================================================

def compute_table_vi_matrix(
    records: List[Dict[str, Any]],
    p0_u_default: float = 4.0,
    target_kinds: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Tính toán ma trận kết cục 4 hàng (ρ_px) x 5 cột (ρ_tok).

    Mỗi ô: % Hallucination / % Lỗi nhận thức / % Từ chối (và N mẫu).
    """
    if target_kinds is not None:
        sub_records = [r for r in records if r.get("kind") in target_kinds]
    else:
        # Mặc định lấy toàn bộ hoặc Q1 + Q3-HF như trong bài báo
        sub_records = records

    row_bins = ["< p0_U", "p0_U - 2p0_U", "2p0_U - 4p0_U", ">= 4p0_U"]
    col_bins = ["< 0.5", "0.5 - 1.0", "1.0 - 2.0", "2.0 - 4.0", ">= 4.0"]

    # cell: [total, hallu, percep_err, abstain, correct]
    grid = {r: {c: {"total": 0, "hallu": 0, "percep_err": 0, "abstain": 0, "correct": 0} for c in col_bins} for r in row_bins}

    for item in sub_records:
        rho_px = float(item.get("rho_px", 10.0))
        p0_u = float(item.get("p0_U_px", p0_u_default))
        rho_tok = float(item.get("rho_tok", 1.0))
        is_ans = item.get("answerable_by_sensor", (rho_px >= p0_u))
        is_correct = bool(item.get("is_correct", False))
        is_abstain = bool(item.get("is_abstain", False))

        # Phân hàng theo rho_px
        if rho_px < p0_u or not is_ans:
            row_key = "< p0_U"
        elif rho_px < 2.0 * p0_u:
            row_key = "p0_U - 2p0_U"
        elif rho_px < 4.0 * p0_u:
            row_key = "2p0_U - 4p0_U"
        else:
            row_key = ">= 4p0_U"

        # Phân cột theo rho_tok
        if rho_tok < 0.5:
            col_key = "< 0.5"
        elif rho_tok < 1.0:
            col_key = "0.5 - 1.0"
        elif rho_tok < 2.0:
            col_key = "1.0 - 2.0"
        elif rho_tok < 4.0:
            col_key = "2.0 - 4.0"
        else:
            col_key = ">= 4.0"

        cell = grid[row_key][col_key]
        cell["total"] += 1

        if row_key == "< p0_U":
            # Vật thể không phân giải được bởi cảm biến (U)
            if is_abstain:
                cell["abstain"] += 1
            else:
                # Trả lời bừa khi ảnh không thể xác định -> Hallucination
                cell["hallu"] += 1
        else:
            # Vật thể phân giải được (A)
            if is_correct:
                cell["correct"] += 1
            elif is_abstain:
                cell["abstain"] += 1  # Từ chối thừa
            else:
                cell["percep_err"] += 1  # Lỗi nhận thức

    # Định dạng kết quả dạng %
    formatted_grid = {}
    for r in row_bins:
        formatted_grid[r] = {}
        for c in col_bins:
            cell = grid[r][c]
            n = cell["total"]
            if n == 0:
                formatted_grid[r][c] = {
                    "text": "— / — / —",
                    "n": 0,
                    "pct_hallu": 0.0,
                    "pct_percep_err": 0.0,
                    "pct_abstain": 0.0,
                    "pct_correct": 0.0
                }
            else:
                pct_h = (cell["hallu"] / n) * 100.0
                pct_pe = (cell["percep_err"] / n) * 100.0
                pct_ab = (cell["abstain"] / n) * 100.0
                pct_cor = (cell["correct"] / n) * 100.0
                formatted_grid[r][c] = {
                    "text": f"{pct_h:.1f}% / {pct_pe:.1f}% / {pct_ab:.1f}%",
                    "n": n,
                    "pct_hallu": pct_h,
                    "pct_percep_err": pct_pe,
                    "pct_abstain": pct_ab,
                    "pct_correct": pct_cor
                }

    return {"grid": formatted_grid, "raw": grid, "row_bins": row_bins, "col_bins": col_bins}


# ==============================================================================
# 2. BẢNG VII: MÔ HÌNH HỒI QUY LOGISTIC KIỂM ĐỊNH H1 (β₁ < 0)
# ==============================================================================

def fit_logistic_regression(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Hồi quy Logistic: logit(P(y=1)) = beta_0 + beta_1 * x.

    Tính toán bằng Maximum Likelihood Estimation với ma trận Fisher Information.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(x)
    if N < 5:
        return {"beta_0": 0.0, "beta_1": 0.0, "se_1": 0.0, "z": 0.0, "p_value": 1.0, "ci_lower": 0.0, "ci_upper": 0.0, "converged": False}

    X = np.column_stack([np.ones_like(x), x])

    def nll_loss(beta):
        logits = np.clip(X @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-12
        loss = -np.sum(y * np.log(np.clip(p, eps, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, eps, 1.0)))
        return loss

    res = minimize(nll_loss, [0.0, 0.0], method="BFGS")
    beta = res.x
    logits = np.clip(X @ beta, -35.0, 35.0)
    p = 1.0 / (1.0 + np.exp(-logits))
    w = p * (1.0 - p)
    H = X.T @ (X * w[:, None])

    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.maximum(0.0, np.diag(cov)))
    except Exception:
        se = np.array([np.nan, np.nan])

    se_1 = float(se[1]) if not np.isnan(se[1]) else 0.0
    z = beta[1] / max(1e-12, se_1)
    # Kiểm định 2 phía
    p_val = float(2.0 * (1.0 - norm.cdf(abs(z))))
    ci_lower = float(beta[1] - 1.96 * se_1)
    ci_upper = float(beta[1] + 1.96 * se_1)

    return {
        "beta_0": float(beta[0]),
        "beta_1": float(beta[1]),
        "se_1": se_1,
        "z": float(z),
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": float(np.exp(beta[1])),
        "converged": bool(res.success),
        "n_samples": N
    }


def compute_table_vii_mixed_effects(
    records: List[Dict[str, Any]],
    only_answerable: bool = True
) -> Dict[str, Any]:
    """Kiểm định H1: logit P(error) = beta_0 + beta_1 * log2(rho_tok).

    Kỳ vọng H1: beta_1 < 0 (khi token footprint tăng thì tỷ lệ lỗi giảm).
    """
    valid_items = []
    for item in records:
        if only_answerable and not item.get("answerable_by_sensor", True):
            continue
        rho_tok = float(item.get("rho_tok", 0.0))
        if rho_tok <= 0:
            continue
        is_err = 0 if bool(item.get("is_correct", False)) else 1
        valid_items.append((math.log2(rho_tok), is_err))

    if not valid_items:
        return {"error": "Không đủ mẫu hợp lệ"}

    x = np.array([item[0] for item in valid_items])
    y = np.array([item[1] for item in valid_items])

    res = fit_logistic_regression(x, y)
    res["h1_supported"] = bool(res["beta_1"] < 0 and res["p_value"] < 0.05)
    return res


# ==============================================================================
# 3. BẢNG VIII: HAI NGƯỠNG PHÂN GIẢI (H2)
# ==============================================================================

def estimate_resolution_threshold_50(
    x_values: np.ndarray,
    is_correct_arr: np.ndarray
) -> Dict[str, Any]:
    """Ước lượng ngưỡng 50% bằng hàm psychometric logistic: logit(acc) = a0 + a1 * log2(x)."""
    if len(x_values) < 10 or np.sum(is_correct_arr) == 0:
        return {"threshold": np.nan, "a0": 0.0, "a1": 0.0}

    log_x = np.log2(np.maximum(1e-3, x_values))
    reg = fit_logistic_regression(log_x, is_correct_arr)

    a0, a1 = reg["beta_0"], reg["beta_1"]
    if a1 <= 1e-6:
        # Nếu đường dốc không dương hoặc quá phẳng
        return {"threshold": np.nan, "a0": a0, "a1": a1}

    # Tại xác suất 50% (logit = 0): a0 + a1 * log2(x_50) = 0 => log2(x_50) = -a0 / a1
    log2_thresh = -a0 / a1
    thresh = float(2.0 ** log2_thresh)

    # Giới hạn vật lý để tránh giá trị phi lý
    if thresh < 0.1 or thresh > 10000.0:
        thresh = np.nan

    return {"threshold": thresh, "a0": a0, "a1": a1, "p_value": reg["p_value"]}


def compute_table_viii_thresholds(
    records: List[Dict[str, Any]],
    p_patch: int = 32
) -> Dict[str, Any]:
    """Tính ngưỡng cảm biến p₀⁵⁰ và ngưỡng token ρ*_tok."""
    # 1. Ước lượng p₀⁵⁰ dưới điều kiện unconstrained token (rho_tok >= 2.0 hoặc >= 4.0)
    unconstrained_tokens = [
        r for r in records
        if float(r.get("rho_tok", 0.0)) >= 2.0 and r.get("answerable_by_sensor", True)
    ]
    if not unconstrained_tokens:
        unconstrained_tokens = [r for r in records if r.get("answerable_by_sensor", True)]

    px_vals = np.array([float(r.get("rho_px", 1.0)) for r in unconstrained_tokens])
    cor_vals = np.array([1.0 if r.get("is_correct") else 0.0 for r in unconstrained_tokens])
    p0_50_res = estimate_resolution_threshold_50(px_vals, cor_vals)

    # 2. Ước lượng ρ*_tok dưới điều kiện unconstrained sensor (rho_px >= 2 * p0_U)
    unconstrained_pixels = [
        r for r in records
        if float(r.get("rho_px", 0.0)) >= 2.0 * float(r.get("p0_U_px", 4.0)) and r.get("answerable_by_sensor", True)
    ]
    if not unconstrained_pixels:
        unconstrained_pixels = [r for r in records if r.get("answerable_by_sensor", True)]

    tok_vals = np.array([float(r.get("rho_tok", 0.1)) for r in unconstrained_pixels])
    cor_tok_vals = np.array([1.0 if r.get("is_correct") else 0.0 for r in unconstrained_pixels])
    rho_tok_star_res = estimate_resolution_threshold_50(tok_vals, cor_tok_vals)

    p0_50 = p0_50_res["threshold"]
    rho_tok_star = rho_tok_star_res["threshold"]
    model_px_equiv = (rho_tok_star * p_patch) if not np.isnan(rho_tok_star) else np.nan

    # Giá trị p0_U trung bình quan sát được trong benchmark
    all_p0_u = [float(r.get("p0_U_px", 4.0)) for r in records]
    avg_p0_u = float(np.mean(all_p0_u)) if all_p0_u else 4.0

    return {
        "P_patch": p_patch,
        "p0_50_px": p0_50,
        "p0_U_px": avg_p0_u,
        "rho_tok_star": rho_tok_star,
        "model_px_threshold": model_px_equiv
    }


# ==============================================================================
# 4. BẢNG IX: BẬC THANG JOHNSON & KIỂM ĐỊNH JONCKHEERE-TERPSTRA (H3)
# ==============================================================================

def jonckheere_terpstra_test(ordered_groups: List[List[float]]) -> Dict[str, Any]:
    """Kiểm định thứ tự đơn điệu Jonckheere-Terpstra cho k nhóm có thứ tự tiên nghiệm:

    H0: theta_1 = theta_2 = ... = theta_k
    H1: theta_1 <= theta_2 <= ... <= theta_k (có ít nhất 1 dấu <).
    """
    k = len(ordered_groups)
    n = [len(g) for g in ordered_groups]
    N = sum(n)
    if k < 2 or any(sz == 0 for sz in n):
        return {"J": 0.0, "z": 0.0, "p_value": 1.0}

    J = 0.0
    for i in range(k):
        g_i = np.asarray(ordered_groups[i], dtype=float)
        for j in range(i + 1, k):
            g_j = np.asarray(ordered_groups[j], dtype=float)
            diff = g_j[:, None] - g_i[None, :]
            u_ij = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
            J += u_ij

    mean_J = (N**2 - sum(ni**2 for ni in n)) / 4.0
    var_J = (N**2 * (2 * N + 3) - sum(ni**2 * (2 * ni + 3) for ni in n)) / 72.0
    std_J = math.sqrt(max(1e-12, var_J))

    z = (J - mean_J) / std_J
    p_val = float(1.0 - norm.cdf(z))

    return {
        "J": float(J),
        "E_J": float(mean_J),
        "std_J": float(std_J),
        "z": float(z),
        "p_value": p_val
    }


def compute_table_ix_johnson_hierarchy(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tính ngưỡng p₀⁵⁰ cho từng loại câu hỏi theo bậc thang Johnson và kiểm định JT."""
    # Thứ tự theo Johnson:
    # 1. Q3-LF (Màu sắc / Tần số thấp)
    # 2. Q1 (Tồn tại / Detection)
    # 3. Q4 (Định hướng / Orientation)
    # 4. Q2 (Đếm / Recognition)
    # 5. Q5 (Phân loại chi tiết / Identification)
    kinds_order = ["Q3-LF", "Q1", "Q4", "Q2", "Q5"]
    kind_labels = {
        "Q3-LF": "Q3-LF Màu sắc (Low-freq)",
        "Q1": "Q1 Tồn tại (Detection)",
        "Q4": "Q4 Hướng OBB (Orientation)",
        "Q2": "Q2 Đếm / Lớp (Recognition)",
        "Q5": "Q5 Loại con (Identification)"
    }

    results_by_kind = {}
    sampled_p0_distributions = []

    for k in kinds_order:
        k_records = [
            r for r in records
            if r.get("kind") == k and r.get("answerable_by_sensor", True)
        ]
        if not k_records:
            results_by_kind[k] = {"threshold": np.nan, "n": 0}
            continue

        px_vals = np.array([float(r.get("rho_px", 1.0)) for r in k_records])
        cor_vals = np.array([1.0 if r.get("is_correct") else 0.0 for r in k_records])

        est = estimate_resolution_threshold_50(px_vals, cor_vals)
        results_by_kind[k] = {
            "threshold": est["threshold"],
            "n": len(k_records),
            "label": kind_labels[k]
        }

        # Lấy phân phối rho_px của các câu trả lời đúng làm proxy cho JT test
        correct_px = [float(r.get("rho_px", 1.0)) for r in k_records if r.get("is_correct")]
        if correct_px:
            sampled_p0_distributions.append(correct_px)

    # Thực hiện Jonckheere-Terpstra nếu có ít nhất 2 nhóm
    jt_res = {"p_value": 1.0, "z": 0.0}
    if len(sampled_p0_distributions) >= 2:
        jt_res = jonckheere_terpstra_test(sampled_p0_distributions)

    return {
        "by_kind": results_by_kind,
        "jonckheere_terpstra": jt_res,
        "monotonic_hierarchy_supported": bool(jt_res.get("p_value", 1.0) < 0.05)
    }


# ==============================================================================
# 5. BẢNG X: DỰ ĐOÁN CÓ CHỌN LỌC (SELECTIVE PREDICTION - H4)
# ==============================================================================

def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Tích phân hình thang tương thích cả NumPy 1.x và 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    elif hasattr(np, "trapz"):
        return float(getattr(np, "trapz")(y, x))
    else:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        return float(np.sum((x_arr[1:] - x_arr[:-1]) * (y_arr[1:] + y_arr[:-1]) / 2.0))


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    is_corrects: np.ndarray
) -> Dict[str, Any]:
    """Tính toán đường cong Risk-Coverage và AURC từ vector confidence và correctness."""
    N = len(confidences)
    if N == 0:
        return {"aurc": 0.0, "cov_at_5": 0.0, "cov_at_10": 0.0, "cov_at_20": 0.0, "empirical_risk_at_10": 0.0}

    # Sắp xếp theo confidence giảm dần
    sort_idx = np.argsort(-confidences)
    sorted_corr = is_corrects[sort_idx]

    cum_correct = np.cumsum(sorted_corr)
    k_range = np.arange(1, N + 1)

    coverages = k_range / N
    risks = (k_range - cum_correct) / k_range  # Tỷ lệ lỗi trong số k mẫu được chấp nhận

    # AURC bằng tích phân hình thang
    aurc = trapz_compat(risks, coverages)

    def get_cov_at_alpha(alpha: float) -> Tuple[float, float]:
        # Tìm coverage lớn nhất sao cho risk <= alpha
        valid_indices = np.where(risks <= alpha)[0]
        if len(valid_indices) == 0:
            return 0.0, float(risks[0])
        max_idx = valid_indices[-1]
        return float(coverages[max_idx] * 100.0), float(risks[max_idx] * 100.0)

    cov_5, _ = get_cov_at_alpha(0.05)
    cov_10, risk_10 = get_cov_at_alpha(0.10)
    cov_20, _ = get_cov_at_alpha(0.20)

    return {
        "aurc": aurc,
        "cov_at_5": cov_5,
        "cov_at_10": cov_10,
        "cov_at_20": cov_20,
        "empirical_risk_at_10": risk_10
    }


def compute_table_x_selective_prediction(
    records: List[Dict[str, Any]],
    p0_u_default: float = 4.0,
    rho_tok_star: float = 2.0
) -> Dict[str, Any]:
    """Đo đạc chỉ số dự đoán có chọn lọc theo Bảng X."""
    # Phân tách tập A (answerable) và U (unanswerable)
    ans_records = [r for r in records if r.get("answerable_by_sensor", True)]
    unans_records = [r for r in records if not r.get("answerable_by_sensor", True)]

    n_ans = len(ans_records)
    n_unans = len(unans_records)
    n_total = len(records)

    methods = {}

    # 1. Zero-shot baseline (chấp nhận toàn bộ)
    zs_acc_ans = (sum(1 for r in ans_records if r.get("is_correct")) / max(1, n_ans)) * 100.0
    zs_risk_ans = 100.0 - zs_acc_ans
    zs_abst_u = (sum(1 for r in unans_records if r.get("is_abstain")) / max(1, n_unans)) * 100.0
    zs_over_abst_a = (sum(1 for r in ans_records if r.get("is_abstain")) / max(1, n_ans)) * 100.0
    zs_sel_acc = ((sum(1 for r in ans_records if r.get("is_correct")) + sum(1 for r in unans_records if r.get("is_abstain"))) / max(1, n_total)) * 100.0

    methods["Zero-shot"] = {
        "requires_internal": "—",
        "extra_calls": 0.0,
        "cov_5": 100.0 if zs_risk_ans <= 5.0 else 0.0,
        "cov_10": 100.0 if zs_risk_ans <= 10.0 else 0.0,
        "cov_20": 100.0 if zs_risk_ans <= 20.0 else 0.0,
        "risk_10": zs_risk_ans,
        "aurc": zs_risk_ans / 100.0,
        "abst_u": zs_abst_u,
        "over_abst_a": zs_over_abst_a,
        "sel_acc": zs_sel_acc
    }

    # 2. Max-logit / Model Confidence
    if n_ans > 0:
        confs_ans = np.array([float(r.get("confidence", 0.5)) for r in ans_records])
        corr_ans = np.array([1.0 if r.get("is_correct") else 0.0 for r in ans_records])
        conf_rc = compute_risk_coverage_curve(confs_ans, corr_ans)
    else:
        conf_rc = {"aurc": 0.0, "cov_at_5": 0.0, "cov_at_10": 0.0, "cov_at_20": 0.0, "empirical_risk_at_10": 0.0}

    methods["Max-logit / Confidence"] = {
        "requires_internal": "✓",
        "extra_calls": 0.0,
        "cov_5": conf_rc["cov_at_5"],
        "cov_10": conf_rc["cov_at_10"],
        "cov_20": conf_rc["cov_at_20"],
        "risk_10": conf_rc["empirical_risk_at_10"],
        "aurc": conf_rc["aurc"],
        "abst_u": zs_abst_u,
        "over_abst_a": 100.0 - conf_rc["cov_at_10"],
        "sel_acc": zs_sel_acc
    }

    # 3. RGSP-abstain (Định tuyến vật lý thuần tuý: rho_px >= p0_U AND rho_tok >= rho_tok_star)
    accepted_ans_rgsp = [
        r for r in ans_records
        if float(r.get("rho_px", 0.0)) >= float(r.get("p0_U_px", p0_u_default))
        and float(r.get("rho_tok", 0.0)) >= rho_tok_star
    ]
    cov_rgsp = (len(accepted_ans_rgsp) / max(1, n_ans)) * 100.0
    corr_rgsp = sum(1 for r in accepted_ans_rgsp if r.get("is_correct"))
    risk_rgsp = ((len(accepted_ans_rgsp) - corr_rgsp) / max(1, len(accepted_ans_rgsp))) * 100.0 if accepted_ans_rgsp else 0.0

    # RGSP từ chối 100% tập U vì rho_px < p0_U
    abst_u_rgsp = 100.0
    over_abst_a_rgsp = 100.0 - cov_rgsp
    sel_acc_rgsp = ((corr_rgsp + n_unans) / max(1, n_total)) * 100.0

    methods["RGSP-abstain (Physical)"] = {
        "requires_internal": "✗",
        "extra_calls": 0.0,
        "cov_5": cov_rgsp if risk_rgsp <= 5.0 else 0.0,
        "cov_10": cov_rgsp if risk_rgsp <= 10.0 else 0.0,
        "cov_20": cov_rgsp if risk_rgsp <= 20.0 else 0.0,
        "risk_10": risk_rgsp,
        "aurc": (risk_rgsp / 100.0) * (cov_rgsp / 100.0),
        "abst_u": abst_u_rgsp,
        "over_abst_a": over_abst_a_rgsp,
        "sel_acc": sel_acc_rgsp
    }

    # 4. RGSP+ (Kết hợp định tuyến vật lý + xếp hạng Confidence nội tại)
    if n_ans > 0:
        hybrid_scores = []
        for r in ans_records:
            is_phys_resolvable = (
                float(r.get("rho_px", 0.0)) >= float(r.get("p0_U_px", p0_u_default))
                and float(r.get("rho_tok", 0.0)) >= rho_tok_star
            )
            base_c = float(r.get("confidence", 0.5))
            score = (1.0 + base_c) if is_phys_resolvable else base_c * 0.1
            hybrid_scores.append(score)

        rgsp_plus_rc = compute_risk_coverage_curve(np.array(hybrid_scores), corr_ans)
    else:
        rgsp_plus_rc = {"aurc": 0.0, "cov_at_5": 0.0, "cov_at_10": 0.0, "cov_at_20": 0.0, "empirical_risk_at_10": 0.0}

    methods["RGSP+ (Physical + Confidence)"] = {
        "requires_internal": "✓",
        "extra_calls": 0.0,
        "cov_5": rgsp_plus_rc["cov_at_5"],
        "cov_10": rgsp_plus_rc["cov_at_10"],
        "cov_20": rgsp_plus_rc["cov_at_20"],
        "risk_10": rgsp_plus_rc["empirical_risk_at_10"],
        "aurc": rgsp_plus_rc["aurc"],
        "abst_u": 100.0,
        "over_abst_a": 100.0 - rgsp_plus_rc["cov_at_10"],
        "sel_acc": sel_acc_rgsp
    }

    return {
        "n_ans": n_ans,
        "n_unans": n_unans,
        "n_total": n_total,
        "methods": methods
    }


# ==============================================================================
# 6. BÁO CÁO TỔNG QUAN & XUẤT MARKDOWN / JSON
# ==============================================================================

def compute_overall_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tổng hợp tỷ lệ đúng theo các khía cạnh: loại câu hỏi, tập dữ liệu, lớp vật thể."""
    total = len(records)
    if total == 0:
        return {}

    correct_cnt = sum(1 for r in records if r.get("is_correct"))
    abstain_cnt = sum(1 for r in records if r.get("is_abstain"))

    by_kind = defaultdict(lambda: {"total": 0, "correct": 0, "abstain": 0})
    for r in records:
        k = r.get("kind", "unknown")
        by_kind[k]["total"] += 1
        if r.get("is_correct"):
            by_kind[k]["correct"] += 1
        if r.get("is_abstain"):
            by_kind[k]["abstain"] += 1

    by_kind_summary = {}
    for k, d in by_kind.items():
        by_kind_summary[k] = {
            "total": d["total"],
            "acc": (d["correct"] / d["total"]) * 100.0,
            "abst_rate": (d["abstain"] / d["total"]) * 100.0
        }

    return {
        "total_evaluated": total,
        "overall_accuracy": (correct_cnt / total) * 100.0,
        "abstention_rate": (abstain_cnt / total) * 100.0,
        "by_kind": by_kind_summary
    }


def format_markdown_report(
    model_name: str,
    overall: Dict[str, Any],
    tbl_vi: Dict[str, Any],
    tbl_vii: Dict[str, Any],
    tbl_viii: Dict[str, Any],
    tbl_ix: Dict[str, Any],
    tbl_x: Dict[str, Any]
) -> str:
    """Tạo báo cáo định dạng Markdown chuẩn GitHub để nộp và lưu trữ."""
    lines = []
    lines.append(f"# Báo Cáo Đánh Giá Khoa Học RS-Solve / RGSP: {model_name}")
    lines.append("")
    lines.append("## 1. Tổng Quan Hiệu Năng Thực Nghiệm")
    lines.append(f"- **Tổng số mẫu đánh giá:** {overall.get('total_evaluated', 0):,} câu")
    lines.append(f"- **Độ chính xác toàn cục (Overall Accuracy):** {overall.get('overall_accuracy', 0.0):.2f}%")
    lines.append(f"- **Tỷ lệ từ chối toàn cục (Abstention Rate):** {overall.get('abstention_rate', 0.0):.2f}%")
    lines.append("")
    lines.append("### Phân bố độ chính xác theo loại câu hỏi (Question Kind):")
    lines.append("| Loại câu hỏi | Mô tả nhiệm vụ | Số mẫu | Độ chính xác (Acc %) | Tỷ lệ từ chối (%) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")

    kind_desc = {
        "Q1": "Tồn tại vật thể (Detection)",
        "Q2": "Đếm số lượng (Counting)",
        "Q3-LF": "Màu sắc vật thể (Low-freq)",
        "Q3-HF": "Chi tiết kết cấu (High-freq)",
        "Q4": "Hướng xoay OBB (Orientation)",
        "Q5": "Nhận dạng loại con (Identification)",
        "Q6": "Định vị không gian (Grounding)"
    }
    for k, stats in sorted(overall.get("by_kind", {}).items()):
        lines.append(f"| **{k}** | {kind_desc.get(k, '—')} | {stats['total']:,} | {stats['acc']:.2f}% | {stats['abst_rate']:.2f}% |")

    lines.append("")
    lines.append("---")
    lines.append("## 2. BẢNG VI: Ma Trận Kết Cục $\\rho_{px} \\times \\rho_{tok}$ (%)")
    lines.append("> *Mỗi ô: % Hallucination / % Lỗi nhận thức / % Từ chối*")
    lines.append("")
    lines.append("| $\\rho_{px} \\backslash \\rho_{tok}$ | < 0.5 | 0.5 - 1.0 | 1.0 - 2.0 | 2.0 - 4.0 | $\\ge$ 4.0 |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    grid = tbl_vi["grid"]
    for r in tbl_vi["row_bins"]:
        row_str = f"| **{r}** |"
        for c in tbl_vi["col_bins"]:
            row_str += f" {grid[r][c]['text']} |"
        lines.append(row_str)

    lines.append("")
    lines.append("---")
    lines.append("## 3. BẢNG VII: Kiểm Định Giả Thuyết Nút Thắt Token $H_1$ (Logistic Regression)")
    lines.append("$$\\text{logit}(P(\\text{error})) = \\beta_0 + \\beta_1 \\log_2(\\rho_{tok})$$")
    lines.append("")
    lines.append(f"- **Hệ số $\\beta_1$ (log $\\rho_{{tok}}$):** `{tbl_vii.get('beta_1', 0.0):.4f}`")
    lines.append(f"- **Khoảng tin cậy 95% (CI 95%):** `[{tbl_vii.get('ci_lower', 0.0):.4f}, {tbl_vii.get('ci_upper', 0.0):.4f}]`")
    lines.append(f"- **Tỷ số chênh (Odds Ratio):** `{tbl_vii.get('odds_ratio', 1.0):.4f}`")
    lines.append(f"- **Giá trị p (p-value):** `{tbl_vii.get('p_value', 1.0):.4e}`")
    h1_status = "✅ **ĐƯỢC XÁC NHẬN** (Hệ số âm có ý nghĩa thống kê $p < 0.05$)" if tbl_vii.get("h1_supported") else "❌ **CHƯA ĐẠT ĐIỀU KIỆN**"
    lines.append(f"- **Kết luận giả thuyết $H_1$:** {h1_status}")

    lines.append("")
    lines.append("---")
    lines.append("## 4. BẢNG VIII: Hai Ngưỡng Phân Giải Độc Lập $H_2$")
    lines.append("")
    lines.append("| Thông số | Ký hiệu | Giá trị thực nghiệm | Đơn vị | Ghi chú |")
    lines.append("| :--- | :---: | :---: | :---: | :--- |")
    lines.append(f"| Kích thước patch hiệu dụng | $P$ | `{tbl_viii.get('P_patch')}` | px | Cấu trúc bộ mã hoá thị giác |")
    lines.append(f"| Ngưỡng cảm biến 50% | $p_0^{{50}}$ | `{tbl_viii.get('p0_50_px', np.nan):.2f}` | px gốc | Đo tại vùng unconstrained tokens |")
    lines.append(f"| Giới hạn vật lý cảm biến | $p_0^U$ | `{tbl_viii.get('p0_U_px', 4.0):.2f}` | px gốc | Ngưỡng dưới không thể giải |")
    lines.append(f"| Ngưỡng token mô hình | $\\rho_{{tok}}^*$ | `{tbl_viii.get('rho_tok_star', np.nan):.2f}` | tokens | Đo tại vùng unconstrained pixels |")
    lines.append(f"| Ngưỡng pixel quy đổi | $\\rho_{{tok}}^* \\cdot P$ | `{tbl_viii.get('model_px_threshold', np.nan):.2f}` | px mô hình | Chuẩn hoá theo kiến trúc |")

    lines.append("")
    lines.append("---")
    lines.append("## 5. BẢNG IX: Bậc Thang Johnson $H_3$ & Kiểm Định Jonckheere-Terpstra")
    lines.append("")
    lines.append("| Cấp bậc nhiệm vụ | Nhiệm vụ | Số mẫu | Ngưỡng cảm biến $p_0^{{50}}$ (px) |")
    lines.append("| :--- | :--- | :---: | :---: |")
    for k, data in tbl_ix.get("by_kind", {}).items():
        th = f"{data['threshold']:.2f}" if not np.isnan(data.get("threshold", np.nan)) else "—"
        lines.append(f"| **{k}** | {data.get('label', k)} | {data.get('n', 0):,} | `{th}` |")

    jt = tbl_ix.get("jonckheere_terpstra", {})
    lines.append("")
    lines.append(f"- **Thống kê kiểm định Jonckheere-Terpstra (J):** `{jt.get('J', 0.0):.1f}` (Kỳ vọng: `{jt.get('E_J', 0.0):.1f}`, z-score: `{jt.get('z', 0.0):.2f}`)")
    lines.append(f"- **p-value:** `{jt.get('p_value', 1.0):.4e}`")
    h3_status = "✅ **ĐƯỢC XÁC NHẬN** (Thứ tự đơn điệu $p_0(\\text{Color}) < p_0(\\text{Det}) < p_0(\\text{Orient}) < p_0(\\text{Recog})$)" if tbl_ix.get("monotonic_hierarchy_supported") else "❌ **CHƯA ĐẠT ĐIỀU KIỆN**"
    lines.append(f"- **Kết luận bậc thang Johnson $H_3$:** {h3_status}")

    lines.append("")
    lines.append("---")
    lines.append("## 6. BẢNG X: Dự Đoán Có Chọn Lọc (Selective Prediction - $H_4$)")
    lines.append("")
    lines.append("| Phương pháp | Cần nội bộ | Gọi thêm | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ | Rủi ro thực @10% | AURC ↓ | Abst.-acc trên U ↑ | Từ chối thừa trên A ↓ | Selective Acc. ↑ |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m_name, m_data in tbl_x.get("methods", {}).items():
        lines.append(
            f"| **{m_name}** | {m_data['requires_internal']} | {m_data['extra_calls']} | "
            f"{m_data['cov_5']:.1f}% | {m_data['cov_10']:.1f}% | {m_data['cov_20']:.1f}% | "
            f"{m_data['risk_10']:.1f}% | {m_data['aurc']:.3f} | {m_data['abst_u']:.1f}% | "
            f"{m_data['over_abst_a']:.1f}% | {m_data['sel_acc']:.1f}% |"
        )

    lines.append("")
    lines.append("---")
    lines.append("*Báo cáo được tính toán tự động bằng Engine Đánh giá Khoa học RGSP/RS-Solve.*")

    return "\n".join(lines)


def evaluate_prediction_file(
    pred_path: Path,
    output_dir: Path,
    p_patch: int = 32
) -> Dict[str, Any]:
    """Tính toán toàn bộ các bảng chỉ số cho 1 file dự đoán."""
    print("=" * 80)
    print(f"BẮT ĐẦU ĐO LƯỜNG CHỈ SỐ KHOA HỌC CHO: {pred_path.name}")
    print("=" * 80)

    records = load_predictions(pred_path)
    if not records:
        print(f"[-] File dự đoán rỗng hoặc không hợp lệ: {pred_path}")
        return {}

    model_name = pred_path.stem.replace("preds_", "").upper()

    overall = compute_overall_summary(records)
    tbl_vi = compute_table_vi_matrix(records)
    tbl_vii = compute_table_vii_mixed_effects(records)
    tbl_viii = compute_table_viii_thresholds(records, p_patch=p_patch)
    tbl_ix = compute_table_ix_johnson_hierarchy(records)
    tbl_x = compute_table_x_selective_prediction(records, p0_u_default=tbl_viii.get("p0_U_px", 4.0))

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Xuất file Markdown
    md_content = format_markdown_report(model_name, overall, tbl_vi, tbl_vii, tbl_viii, tbl_ix, tbl_x)
    md_file = output_dir / f"eval_report_{pred_path.stem}.md"
    md_file.write_text(md_content, encoding="utf-8")
    print(f"[+] Đã tạo báo cáo Markdown tại: {md_file}")

    # 2. Xuất file JSON
    full_data = {
        "model_name": model_name,
        "overall": overall,
        "table_vi": tbl_vi,
        "table_vii": tbl_vii,
        "table_viii": tbl_viii,
        "table_ix": tbl_ix,
        "table_x": tbl_x
    }
    json_file = output_dir / f"eval_report_{pred_path.stem}.json"
    json_file.write_text(json.dumps(full_data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[+] Đã lưu cấu trúc JSON tại: {json_file}")

    # In tóm tắt ra màn hình
    print("\n" + "=" * 80)
    print(f"TỔNG KẾT KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH: {model_name}")
    print("=" * 80)
    print(f"  • Số lượng mẫu đã chạy  : {overall.get('total_evaluated', 0):,} câu")
    print(f"  • Độ chính xác toàn cục  : {overall.get('overall_accuracy', 0.0):.2f}%")
    print(f"  • Hệ số β₁ (H1)         : {tbl_vii.get('beta_1', 0.0):.4f} (p-value: {tbl_vii.get('p_value', 1.0):.4e})")
    print(f"  • Ngưỡng cảm biến p₀⁵⁰  : {tbl_viii.get('p0_50_px', np.nan):.2f} px")
    print(f"  • Ngưỡng token ρ*_tok    : {tbl_viii.get('rho_tok_star', np.nan):.2f} tokens")
    print(f"  • Kiểm định Johnson H3  : z = {tbl_ix.get('jonckheere_terpstra', {}).get('z', 0.0):.2f} (p = {tbl_ix.get('jonckheere_terpstra', {}).get('p_value', 1.0):.4e})")
    cov10_rgsp = tbl_x.get("methods", {}).get("RGSP-abstain (Physical)", {}).get("cov_10", 0.0)
    print(f"  • Cov@10% RGSP-abstain   : {cov10_rgsp:.1f}%")
    print("=" * 80 + "\n")

    return full_data


def main():
    parser = argparse.ArgumentParser(description="Tính toán chỉ số khoa học cho Benchmark RS-Solve (Tables VI - X)")
    parser.add_argument("--preds-file", required=True, nargs="+", help="Đường dẫn đến 1 hoặc nhiều file dự đoán jsonl")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent.parent / "eval_reports"), help="Thư mục xuất báo cáo")
    parser.add_argument("--p-patch", type=int, default=32, help="Kích thước patch P của mô hình (32 cho Qwen2.5-VL, 28 cho InternVL, 14 cho LLaVA)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    for p_file in args.preds_file:
        pf = Path(p_file)
        if not pf.exists():
            print(f"[-] Không tìm thấy file: {pf}")
            continue
        evaluate_prediction_file(pf, out_dir, p_patch=args.p_patch)


if __name__ == "__main__":
    main()
