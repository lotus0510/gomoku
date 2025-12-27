# 文檔索引

本文檔提供專案所有文檔的組織結構和快速導覽。

---

## 📚 快速導覽

### 新用戶起步
1. [README.md](../README.md) - 專案概述和快速開始
2. [SETUP_GUIDE.md](guides/SETUP_GUIDE.md) - 環境設置詳細指南
3. [TRAINING_CONCEPTS.md](guides/TRAINING_CONCEPTS.md) - 訓練原理和概念

### 開發者
1. [DEVELOPER_GUIDE.md](guides/DEVELOPER_GUIDE.md) - 代碼結構和開發指南
2. [MONITORING_METRICS.md](guides/MONITORING_METRICS.md) - 訓練監控指標
3. [analysis/README.md](../analysis/README.md) - 分析工具使用說明

### 問題排查
1. [CRITICAL_BUG_FIX.md](troubleshooting/CRITICAL_BUG_FIX.md) - MCTS 根節點 BUG 修復
2. [WHY_NO_DEFENSE.md](troubleshooting/WHY_NO_DEFENSE.md) - 防守策略問題分析
3. [TRAINING_FAILURE_REPORT.md](troubleshooting/TRAINING_FAILURE_REPORT.md) - 訓練故障診斷報告
4. [TRAINING_METRICS_GUIDE.md](troubleshooting/TRAINING_METRICS_GUIDE.md) - 訓練指標參考指南

### 高級策略
1. [CUSTOM_RESET_GUIDE.md](guides/CUSTOM_RESET_GUIDE.md) ⭐ - 自定義重置策略（推薦）
2. [AUTO_ADJUSTMENT_GUIDE.md](guides/AUTO_ADJUSTMENT_GUIDE.md) - 自動參數調整系統

---

## 📁 文檔結構

```
docs/
├── INDEX.md                             # 本文件 - 文檔索引
├── guides/                              # 使用指南
│   ├── SETUP_GUIDE.md                  # 環境設置指南
│   ├── TRAINING_CONCEPTS.md            # 訓練概念和機制
│   ├── DEVELOPER_GUIDE.md              # 開發者指南
│   ├── MONITORING_METRICS.md           # 監控指標說明
│   ├── CUSTOM_RESET_GUIDE.md           # ⭐ 自定義重置策略（推薦）
│   └── AUTO_ADJUSTMENT_GUIDE.md        # 自動參數調整系統
├── troubleshooting/                     # 問題排查
│   ├── CRITICAL_BUG_FIX.md             # 關鍵 BUG 修復報告
│   ├── WHY_NO_DEFENSE.md               # 防守問題分析
│   ├── TRAINING_FAILURE_REPORT.md      # 訓練故障診斷報告
│   ├── CONFIG_INVESTIGATION_REPORT.md  # 配置調查報告
│   └── TRAINING_METRICS_GUIDE.md       # 訓練指標參考指南
└── archived/                            # 已歸檔文檔
    ├── FILE_REORGANIZATION_PLAN.md     # 文件重組計劃
    ├── CLEANUP_REPORT.md               # 清理報告
    ├── FINAL_RECOMMENDATION.md         # 最終建議
    ├── LOG.md                          # 變更日誌
    ├── TRAINING_IMPROVEMENT_PLAN.md    # 訓練改進計劃
    ├── RESTART_TRAINING_GUIDE.md       # 重啟訓練指南（已過時）
    ├── EARLY_TRAINING_ANALYSIS.md      # 早期訓練分析
    └── CLEANUP_SUMMARY_2025-12-23.md   # 清理總結
```

---

## 📖 詳細文檔說明

### 根目錄

#### [README.md](../README.md)
**專案主文檔**
- 專案簡介和特性
- 快速開始指南
- 訓練流程概述
- 配置參數詳解
- 監控指標說明
- 常見問題

**適合**：所有用戶，首次閱讀

---

### guides/ - 使用指南

#### [SETUP_GUIDE.md](guides/SETUP_GUIDE.md)
**環境設置完整指南**
- 系統需求檢查
- Python 和 PyTorch 安裝
- 依賴安裝和驗證
- 常見問題解決
- GPU/CUDA 配置
- Docker 部署（進階）

**適合**：新用戶首次設置環境

#### [TRAINING_CONCEPTS.md](guides/TRAINING_CONCEPTS.md)
**訓練原理和機制深入解析**
- 強化學習核心概念
- 獎勵機制設計哲學
- VALUE_GAMMA 參數詳解
- MCTS 與價值估計
- 訓練流程詳解
- 常見誤區解答

**適合**：想深入理解訓練機制的用戶

#### [DEVELOPER_GUIDE.md](guides/DEVELOPER_GUIDE.md)
**開發者完整指南**
- 專案架構概述
- 代碼結構說明
- 核心模組詳解
- 擴展和修改指南
- 測試和調試
- 貢獻指南

**適合**：需要修改或擴展代碼的開發者

#### [MONITORING_METRICS.md](guides/MONITORING_METRICS.md)
**訓練監控指標說明**
- 六大核心監控指標
- 指標健康範圍
- 異常診斷方法
- 可視化工具使用

**適合**：監控訓練進度的用戶

#### [CUSTOM_RESET_GUIDE.md](guides/CUSTOM_RESET_GUIDE.md) ⭐
**自定義重置策略完整指南（推薦）**
- 防止價值網絡崩潰
- 快速開始（15分鐘整合）
- 核心機制和觸發條件
- 整合步驟和配置說明
- 監控與調試方法
- vs原方案對比分析

**適合**：遇到價值崩潰問題（MAE從0.77→1.04）的用戶
**優先級**：⭐⭐⭐⭐⭐ 高度推薦

#### [AUTO_ADJUSTMENT_GUIDE.md](guides/AUTO_ADJUSTMENT_GUIDE.md)
**自動參數調整系統**
- 通用停滯檢測機制
- 三級調整策略
- 與自定義重置策略配合使用
- 監控與配置選項

**適合**：需要綜合性停滯檢測的用戶
**優先級**：⭐⭐⭐ 作為補充方案

---

### troubleshooting/ - 問題排查

#### [CRITICAL_BUG_FIX.md](troubleshooting/CRITICAL_BUG_FIX.md)
**MCTS 根節點 BUG 修復報告**
- 問題描述：價值預測全為 0
- 根本原因分析
- 修復方案和代碼
- 影響範圍評估
- 驗證方法

**適合**：遇到價值網路問題的用戶

**狀態**：✅ 已修復（2025-12-23）

#### [WHY_NO_DEFENSE.md](troubleshooting/WHY_NO_DEFENSE.md)
**防守策略缺失問題分析**
- 問題現象
- 可能原因
- 診斷方法
- 解決方案

**適合**：遇到 AI 不防守問題的用戶

#### [TRAINING_FAILURE_REPORT.md](troubleshooting/TRAINING_FAILURE_REPORT.md)
**訓練故障診斷報告**
- 價值網絡崩潰分析
- 配置問題檢查
- MCTS參數驗證
- 詳細故障排查步驟

**適合**：訓練出現異常的用戶

#### [CONFIG_INVESTIGATION_REPORT.md](troubleshooting/CONFIG_INVESTIGATION_REPORT.md)
**配置調查報告**
- MCTS參數實現驗證
- 配置一致性檢查
- 代碼分析結果

**適合**：需要驗證配置的開發者

#### [TRAINING_METRICS_GUIDE.md](troubleshooting/TRAINING_METRICS_GUIDE.md)
**訓練指標參考指南**
- 健康訓練指標基準
- 常見問題診斷
- 參數調整建議
- 實時檢查清單

**適合**：需要監控和診斷訓練的所有用戶
**優先級**：⭐⭐⭐⭐ 重要參考

---

### archived/ - 已歸檔文檔

這些文檔已過時或已被整合到其他文檔中，保留作為歷史參考：

- **FILE_REORGANIZATION_PLAN.md** - 早期文件重組計劃
- **CLEANUP_REPORT.md** - 代碼清理報告
- **FINAL_RECOMMENDATION.md** - 早期最終建議
- **LOG.md** - 舊版變更日誌
- **TRAINING_IMPROVEMENT_PLAN.md** - 早期訓練改進計劃
- **RESTART_TRAINING_GUIDE.md** - 重啟訓練指南（問題已修復）
- **EARLY_TRAINING_ANALYSIS.md** - 早期訓練分析（特定迭代數據）
- **CLEANUP_SUMMARY_2025-12-23.md** - 文檔清理總結

**狀態**：僅供參考，可能包含過時信息

---

## 🔧 分析工具文檔

### [analysis/README.md](../analysis/README.md)
分析工具集使用說明

### 核心分析腳本

位於 `analysis/` 目錄：

| 腳本 | 功能 | 使用場景 |
|-----|------|---------|
| `check_training.py` | ✨ 統一檢查工具 | 學習率/損失/權重/價值網路檢查 |
| `diagnose_training.py` | 訓練健康度診斷 | 定期檢查訓練狀態 |
| `plot_comprehensive.py` | 生成完整分析圖表 | 視覺化訓練進度 |
| `analyze_policy.py` | 策略損失分析 | 診斷策略網路問題 |
| `analyze_sample_efficiency.py` | 樣本效率分析 | 評估訓練數據量 |
| `analyze_progress.py` | 訓練進度分析 | 判斷是否需要繼續訓練 |
| `value_gamma_analysis.py` | VALUE_GAMMA 分析 | 分析時間折扣參數 |
| `monitor_training.py` | 實時訓練監控 | 監控正在進行的訓練 |
| `visualize_game.py` | 遊戲可視化 | 查看對局詳情 |

**使用示例**：
```bash
# ✨ 統一檢查工具（新）
python analysis/check_training.py              # 執行所有檢查
python analysis/check_training.py --check lr   # 僅檢查學習率
python analysis/check_training.py --check loss # 僅檢查損失

# 完整診斷
python analysis/diagnose_training.py

# 生成分析圖表
python analysis/plot_comprehensive.py
```

---

## 🔍 如何找到需要的文檔？

### 按使用場景

| 場景 | 推薦文檔 |
|-----|---------|
| 第一次使用專案 | README.md → SETUP_GUIDE.md |
| 開始訓練前 | README.md → TRAINING_CONCEPTS.md |
| 訓練中遇到問題 | MONITORING_METRICS.md → troubleshooting/ |
| 價值網絡崩潰 | ⭐ CUSTOM_RESET_GUIDE.md |
| 訓練停滯 | AUTO_ADJUSTMENT_GUIDE.md 或 CUSTOM_RESET_GUIDE.md |
| 想修改代碼 | DEVELOPER_GUIDE.md |
| 理解訓練原理 | TRAINING_CONCEPTS.md |
| 診斷訓練問題 | TRAINING_METRICS_GUIDE.md → troubleshooting/ |

### 按問題類型

| 問題 | 解決方案文檔 |
|-----|------------|
| 安裝失敗 | SETUP_GUIDE.md - 常見問題 |
| CUDA 不可用 | SETUP_GUIDE.md - Q1 |
| 內存不足 | SETUP_GUIDE.md - Q2 |
| 價值MAE崩潰（>0.95） | ⭐ CUSTOM_RESET_GUIDE.md |
| 策略損失平穩 | AUTO_ADJUSTMENT_GUIDE.md + analyze_policy.py |
| 價值預測為 0 | CRITICAL_BUG_FIX.md |
| AI 不防守 | WHY_NO_DEFENSE.md |
| 訓練不穩定 | TRAINING_METRICS_GUIDE.md |
| 勝率失衡 | TRAINING_METRICS_GUIDE.md + diagnose_training.py |
| 不知道指標是否正常 | TRAINING_METRICS_GUIDE.md |

### 按用戶類型

**新手用戶**：
1. README.md
2. SETUP_GUIDE.md
3. TRAINING_CONCEPTS.md（可選）
4. TRAINING_METRICS_GUIDE.md（開始訓練後）

**遇到價值崩潰問題的用戶**：
1. ⭐ CUSTOM_RESET_GUIDE.md（必讀）
2. TRAINING_METRICS_GUIDE.md
3. TRAINING_FAILURE_REPORT.md

**進階用戶**：
1. MONITORING_METRICS.md
2. TRAINING_METRICS_GUIDE.md
3. AUTO_ADJUSTMENT_GUIDE.md
4. analysis/ 工具集

**開發者**：
1. DEVELOPER_GUIDE.md
2. 代碼內註釋
3. troubleshooting/ 技術報告

---

## 📝 文檔維護

### 更新頻率
- **README.md**: 每次重大功能更新
- **guides/**: 季度審查
- **troubleshooting/**: 發現問題時添加
- **archived/**: 僅歸檔，不更新

### 貢獻指南
如需修改文檔：
1. 遵循現有格式風格
2. 確保中文流暢性
3. 添加清晰的示例
4. 更新本索引文件

### 報告問題
發現文檔問題請：
1. 創建 Issue 描述問題
2. 註明文檔名稱和位置
3. 建議改進方案（可選）

---

## 📅 最後更新

**日期**：2025-12-26
**版本**：3.0 - 大規模文檔整合與優化

**主要變更**：
- ✅ **新增自定義重置策略指南** - 針對價值崩潰問題的完整解決方案
- ✅ **新增自動調整系統指南** - 通用停滯檢測補充方案
- ✅ **新增訓練指標參考指南** - 健康指標基準和診斷流程
- ✅ **合併重複文檔** - 從21個根目錄文檔減少到1個（README.md）
  - 4份自定義重置策略文檔 → CUSTOM_RESET_GUIDE.md
  - 2份自動調整文檔 → AUTO_ADJUSTMENT_GUIDE.md
  - 5份性能分析文檔 → TRAINING_METRICS_GUIDE.md
- ✅ **移動文檔到正確位置**
  - TRAINING_FAILURE_REPORT.md → troubleshooting/
  - CONFIG_INVESTIGATION_REPORT.md → troubleshooting/
  - 過時文檔 → archived/
- ✅ **刪除完全重複的文檔** - DETAILED_CODE_ANALYSIS.md等
- ✅ **清理備份文件** - 刪除舊備份，保留最新版本
- ✅ **更新文檔索引** - 反映新結構，添加優先級標記

**文檔數量變化**：
- 根目錄: 21個.md → 1個（README.md）
- docs/guides/: 4個 → 6個（新增2個重要指南）
- docs/troubleshooting/: 2個 → 5個（新增3個診斷文檔）
- docs/archived/: 5個 → 8個（歸檔3個過時文檔）

---

**需要幫助？**
- 查看 [README.md](../README.md) 開始
- 閱讀 [SETUP_GUIDE.md](guides/SETUP_GUIDE.md) 設置環境
- 運行 `python analysis/diagnose_training.py` 診斷問題
