# プロジェクトの概要

録画映像から車速を検出・推定し、制限速度超過を検知する車両解析アプリケーションです。
YOLOv8による車検出、SORTによる追跡、4点ホモグラフィによる距離キャリブレーション、
EasyOCRによるナンバープレート認識機能を搭載しています。

## 主な機能

- 車検出と追跡: YOLOv8（車、バイク、バス、トラック対応）とSORTアルゴリズムを使用
- リアルタイム速度推定: 4点ホモグラフィ変換とフレーム差分によるピクセル→メートル変換
- 速度超過警告: 設定閾値（デフォルト60 km/h）を超える車を赤色で表示
- ナンバープレート認識: YOLOv8による検出とEasyOCRによるOCR処理
- キャリブレーション機能:
  - 4点ホモグラフィ設定（画像上の四角形を実寸に対応させる）
  - 進行方向指定（速度ベクトル計算）
  - ドラッグによるポイント微調整

## ディレクトリ構成


















## システムアーキテクチャ

```
main.py (メインアプリ)
├── detector_yolo.py (YOLO車検出)
├── tracker_sort.py (SORT追跡)
├── speed_estimator.py (速度推定)
├── plate_ocr.py (ナンバープレートOCR)
├── calibration.py (4点ホモグラフィ・進行方向管理)
├── overlay_renderer.py (描画エンジン)
├── video_player.py (ビデオプレイヤー)
├── click_selector.py (マウス選択UI)
├── plate_dialog.py (プレート認識ダイアログ)
├── device_manager.py (GPU/CPU管理)
└── config.py (全体設定)
```

## ファイル構成

| ファイル | 説明 |
|---------|------|
| `main.py` | GUI・イベントハンドリング・メインループ |
| `config.py` | パス設定、YOLO設定、CUDA/CPU設定、描画設定、UI定数 |
| `detector_yolo.py` | YOLOv8モデルロード、車・ナンバープレート検出 |
| `tracker_sort.py` | SORT追跡アルゴリズム、静止車両フィルタ |
| `speed_estimator.py` | ホモグラフィ変換による速度計算、スムージング |
| `plate_ocr.py` | EasyOCR/Tesseract によるナンバープレート認識 |
| `calibration.py` | 4点ホモグラフィ、進行方向の管理・UI処理 |
| `overlay_renderer.py` | BBOXや速度数値の画面描画 |
| `video_player.py` | ビデオ再生、フレーム管理、シーク処理 |
| `click_selector.py` | マウスによるトラックID選択UI |
| `plate_dialog.py` | ナンバープレート情報ダイアログ |
| `device_manager.py` | GPU/CPU自動判別 |
| `sort.py` | SORT追跡アルゴリズムの実装 |

## インストール

### 前提条件

- Python 3.8以上
- CUDA対応GPU（オプション：CPU環境でも動作）
- Windows環境（`config.py` のTesseract設定がWindows向け）

### セットアップ手順

#### 1. 仮想環境の作成・有効化

```bash
# Conda推奨
conda create -n car-speeding-env python=3.10
conda activate car-speeding-env

# または virtualenv
python -m venv venv
./venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/Mac
```

#### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

主要な依存パッケージ：
- `ultralytics` - YOLOv8
- `opencv-python` - 画像処理
- `torch` - YOLO推論エンジン
- `easyocr` - ナンバープレート認識（推奨）
- `numpy` - 数値計算
- `pillow` - 画像操作

#### 3. YOLOモデルのダウンロード

`models/` ディレクトリに以下を配置：

```bash
models/
├── yolov8s.pt              # 車検出用（大：精度重視）
├── yolov8n-np.pt           # ナンバープレート検出用
└── EDSR_x2.pb              # 画像超解像（オプション）
```

**ダウンロード：**
```bash
cd models
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"
python -c "from ultralytics import YOLO; YOLO('yolov8n-np.pt')"
cd ..
```

## 使い方

### 基本操作

```bash
python main.py
```

を実行するとGUIウィンドウが起動します。

### UI操作ガイド

| 操作 | 説明 |
|-----|------|
| **映像選択ボタン** | ビデオファイルを選択して読み込み |
| **Play/Pause** | 再生/一時停止 |
| **シークバー**（上部灰色） | フレーム位置をドラッグして移動 |
| **[□ 車BBOX]** チェックボックス | 車の検出枠表示を切り替え |
| **[□ ナンバープレートBBOX]** | ナンバープレート枠表示を切り替え |
| **[▢ 4点キャリブレーション]** トグル | 4点指定モード開始（画像上を4回クリック） |
| **[▢ 進行方向]** トグル | 進行方向指定モード開始（画像上を2回クリック） |
| **幅 [m]** テキストボックス | 4点で指定した四角形の幅を入力 |
| **奥行き [m]** テキストボックス | 4点で指定した四角形の奥行きを入力 |

### 設定ファイル（config.py）

主要な設定項目：

```python
# YOLO設定
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # 車、バイク、バス、トラック
CONF_TH = 0.20                    # 検出感度（低いほど検出多い）

# 速度判定
SPEED_THRESHOLD_KMH = 60          # 超過警告閾値 [km/h]
SPEED_SMOOTHING_WINDOW = 15       # 速度スムージングのフレーム数
SPEED_MIN_THRESHOLD_KMH = 3.0     # 停車判定閾値 [km/h]

# UI表示
MAX_DISPLAY_WIDTH = 1280          # 最大表示幅 [px]
MAX_DISPLAY_HEIGHT = 720          # 最大表示高さ [px]

# GPU/CPU
USE_GPU = True                    # GPU使用フラグ
```

## 技術仕様

### 車検出（detector_yolo.py）

- **モデル**: YOLOv8s（精度重視）+ YOLOv8n-np（ナンバープレート）
- **入力**: BGR画像（OpenCV形式）
- **出力**: BBox座標 `(x1, y1, x2, y2)` と信頼度スコア
- **対象クラス**: COCO車、バイク、バス、トラック（ID: 2, 3, 5, 7）

### 追跡（tracker_sort.py）

- **アルゴリズム**: Simple Online and Realtime Tracking (SORT)
- **動作原理**: IoU距離による関連付け + Kalmanフィルタ
- **出力**: トラックID（車ごとに固有ID）

### 速度推定（speed_estimator.py）

**計算フロー:**
1. BBoxの底辺中央を「タイヤ接地点」として使用
2. フレーム間のピクセル移動量を取得
3. **4点ホモグラフィ変換**: `img_px → real_world_m`
4. ts フレーム分の移動距離 → 秒速 → km/h に変換
5. **スムージング**: 直近15フレームの移動距離の中央値を使用

**特徴:**
- 動画時刻（フレーム数 / fps）をベースに計算 → 処理遅延の影響を除去
- シーク検知で履歴自動クリア → 速度爆発防止
- ピクセル移動量が3px以下 → ノイズとして除外
- 3 km/h以下 → 停車判定

### ナンバープレート認識（plate_ocr.py）

- **検出**: YOLOv8n-np（ナンバープレート専用モデル）
- **認識**:
  - **推奨**: EasyOCR（日本語精度高、GPU対応）
  - **軽量版**: Tesseract OCR（CPU、セットアップ簡単）
- **対応言語**: 日本語・英語

### キャリブレーション（calibration.py）

**4点ホモグラフィ方式:**
- ユーザーが映像上で道路上の矩形4隅をクリック
- 実際の幅×奥行き（単位m）を入力
- OpenCVの `cv2.getPerspectiveTransform()` + `cv2.perspectiveTransform()` で変換
- 画像座標 ↔ 実世界座標を相互変換可能

**進行方向:**
- 2クリックで方向ベクトルを指定
- 速度成分の計算（カメラに対する垂直成分を使用）

## パフォーマンス

- **推論速度**: GPU環境で 15-30 FPS（ビデオサイズ依存）
- **メモリ使用**: ~2-3 GB（YOLOv8s + Kalman × 複数車両）
- **推奨スペック**:
  - GPU: NVIDIA CUDA対応（RTX 3060以上推奨）
  - RAM: 8 GB以上
  - CPU: Core i5 / AMD Ryzen 5以上

## トラブルシューティング

### Q. GUIが起動しない
**A.** `USE_GPU = False` に変更（config.py）してCPU推論に切り替えて試してください。

### Q. ナンバープレートが認識されない
**A.** EasyOCRの言語パラメータを確認してください（plate_ocr.py）

### Q. 速度値がおかしい
**A.** キャリブレーション設定を確認してください：
- 4点が実際の矩形の隅になっているか
- 幅・奥行きのメートル値が正確か
- フレームレート（fps）が正しく取得されているか

### Q. 車が追跡できない
**A.** 
- `CONF_TH` を下げて検出感度を上げてください（0.1～0.15推奨）
- ビデオ品質が低い場合はYOLOv8mをお試しください
```

## 既知の制限

- **複数台同時**: 現在の実装は5～10台程度まで安定
- **逆光時**: 検出精度が低下する場合があります
- **高速移動**: フレームレート不足で追跡ロスの可能性
- **Tesseract**: 日本語ナンバープレートの認識精度がEasyOCRより低い

## ライセンス

??

## 作成日

2026年3月