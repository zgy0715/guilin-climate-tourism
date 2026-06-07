#!/usr/bin/env python3
"""将 数据/ 目录下所有 CSV 文件转为 Excel 格式"""
import pandas as pd
import os

csv_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "数据")
)

for f in os.listdir(csv_dir):
    if f.endswith('.csv'):
        csv_path = os.path.join(csv_dir, f)
        xlsx_path = os.path.join(csv_dir, f.replace('.csv', '.xlsx'))
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            df.to_excel(xlsx_path, index=False, engine='openpyxl')
            print(f'OK: {f} -> {f.replace(".csv", ".xlsx")}')
        except Exception as e:
            print(f'FAIL: {f} - {e}')

print('Done!')
