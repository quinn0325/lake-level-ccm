"""把附录表汇总成单一 Excel 工作簿：每张表一个工作表，外加一页目录。

依赖 build_appendices.py 已生成的 CSV（appendices/*.csv）与其中记录的表题、
说明。表题与说明写在每个工作表的前两行，第三行为表头并冻结，便于滚动查看。

跑法
----
    python code/07_tables/build_appendices.py        # 先生成 CSV
    python code/07_tables/build_appendices_xlsx.py   # 再汇总成 xlsx

输出
----
    final/appendices/全部附录.xlsx
"""
import re
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = Path(__file__).resolve().parents[2] / "appendices"
XLSX = OUT / "全部附录.xlsx"

SECTIONS = [
    ("附录A　数据与数据可用性",
     ["A1_lakes_and_stations", "A2_data_availability",
      "A3_flagged_outliers"]),
    ("附录B　CCM 分析",
     ["B1_embedding_parameters", "B2_supported_drivers",
      "B3_supported_between_lake_edges", "B4_lake_pair_strength"]),
    ("附录C　预测分析",
     ["C1_selected_predictors", "C2_model_configurations",
      "C3_xgboost_hyperparameters", "C4_dm_summary"]),
]

HEAD_FILL = PatternFill("solid", fgColor="DCE6F1")
SEC_FILL = PatternFill("solid", fgColor="F2F2F2")
THIN = Side(style="thin", color="BFBFBF")


def meta(stem):
    """从同名 .md 里取出表题与说明。"""
    lines = (OUT / f"{stem}.md").read_text().split("\n")
    title = lines[0].strip("* ")
    note = " ".join(l.strip() for l in lines[1:] if l.strip()
                    and not l.startswith("|"))
    return title, note


def main():
    order, index_rows = [], []
    for sec, stems in SECTIONS:
        for stem in stems:
            title, note = meta(stem)
            df = pd.read_csv(OUT / f"{stem}.csv")
            sheet = re.match(r"([ABC]\d)", stem).group(1)      # A1 / B2 / C5
            order.append((sheet, title, note, df))
            index_rows.append({"附录": sec, "工作表": sheet, "表题": title,
                               "行数": len(df), "列数": df.shape[1]})

    with pd.ExcelWriter(XLSX, engine="openpyxl") as xw:
        pd.DataFrame(index_rows).to_excel(xw, sheet_name="目录", index=False,
                                          startrow=2)
        for sheet, title, note, df in order:
            df.to_excel(xw, sheet_name=sheet, index=False, startrow=2)

        wb = xw.book
        # ---- 目录页 ----
        ws = wb["目录"]
        ws["A1"] = "附录表索引"
        ws["A1"].font = Font(bold=True, size=13)
        ws["A2"] = ("全部内容来自 2026-08-31 重跑结果；"
                    "生成脚本 code/07_tables/build_appendices.py")
        ws["A2"].font = Font(size=9, color="595959")
        for c in range(1, 6):
            ws.cell(row=3, column=c).font = Font(bold=True)
            ws.cell(row=3, column=c).fill = HEAD_FILL
        for w, col in zip((18, 10, 46, 8, 8), "ABCDE"):
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "A4"

        # ---- 各数据表 ----
        for sheet, title, note, df in order:
            ws = wb[sheet]
            ws["A1"] = title
            ws["A1"].font = Font(bold=True, size=12)
            ws["A2"] = note
            ws["A2"].font = Font(size=9, color="595959")
            ws["A2"].alignment = Alignment(wrap_text=False)
            for c in range(1, df.shape[1] + 1):
                cell = ws.cell(row=3, column=c)
                cell.font = Font(bold=True)
                cell.fill = HEAD_FILL
                cell.border = Border(bottom=THIN)
                cell.alignment = Alignment(horizontal="center")
            for i, col in enumerate(df.columns, start=1):
                longest = max([len(str(col))] +
                              [len(str(v)) for v in df[col].head(200)])
                ws.column_dimensions[get_column_letter(i)].width = \
                    min(max(longest + 2, 9), 46)
            ws.freeze_panes = "A4"
            ws.auto_filter.ref = (f"A3:{get_column_letter(df.shape[1])}"
                                  f"{len(df) + 3}")

    print(f"wrote {XLSX}")
    for sheet, title, _, df in order:
        print(f"  {sheet:<4} {title[:34]:<36} {len(df):>4} 行 × {df.shape[1]} 列")


if __name__ == "__main__":
    main()
