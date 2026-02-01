# plate_ocr.py
# ナンバープレートOCR（EasyOCR版）

import cv2
import numpy as np
import re

try:
    import easyocr
except ImportError:
    raise ImportError("easyocr が必要です。pip install easyocr")


class PlateOCR:
    def __init__(self):
        # EasyOCR初期化（日本語と英語）
        self.reader = easyocr.Reader(['ja', 'en'], gpu=True)
        self.results = {}
        self.plate_images = {}  # ナンバープレートのクロップ画像を保持
    
    def preprocess_plate(self, crop, scale=3):
        """
        ナンバープレート画像の前処理（OCR精度向上のため必須）
        1. 拡大（2〜4倍）
        2. グレースケール
        3. コントラスト強調（CLAHE）
        4. 軽い二値化（適応的閾値）
        """
        if crop is None or crop.size == 0:
            return None
        
        h, w = crop.shape[:2]
        if w == 0 or h == 0:
            return None
        
        # 1. 拡大（3倍）- INTER_CUBIC で高品質拡大
        upscaled = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        
        # 2. グレースケール変換
        if len(upscaled.shape) == 3:
            gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        else:
            gray = upscaled
        
        # 3. コントラスト強調（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 4. 軽い二値化（適応的閾値）
        # ガウシアンぼかしでノイズ除去
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 適応的二値化（局所的な明るさに対応）
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # ブロックサイズ
            2    # 定数C
        )
        
        return binary
    
    def parse_japanese_plate(self, texts):
        """
        OCR結果から日本のナンバープレート形式を解析
        形式: [地域名] [分類番号] [ひらがな] [一連番号]
        例: 品川 300 あ 12-34
        """
        if not texts:
            return {
                "region": "",
                "class_number": "",
                "hiragana": "",
                "plate_number": "",
                "raw": ""
            }
        
        # 全テキストを結合
        full_text = " ".join(texts)
        
        # ひらがな抽出
        hiragana_pattern = r'[あ-んア-ン]'
        hiragana_matches = re.findall(hiragana_pattern, full_text)
        hiragana = hiragana_matches[0] if hiragana_matches else ""
        
        # 数字抽出（ハイフンも含む）
        number_pattern = r'[\d\-・ー]{1,}'
        number_matches = re.findall(number_pattern, full_text)
        
        # 分類番号（1-3桁）と一連番号（4桁）を分離
        class_number = ""
        plate_number = ""
        
        for num in number_matches:
            clean_num = re.sub(r'[\-・ー]', '', num)
            if len(clean_num) <= 3 and not class_number:
                class_number = num
            elif len(clean_num) >= 2:
                plate_number = num
        
        # 地域名抽出（漢字2-4文字）
        region_pattern = r'[一-龥]{2,4}'
        region_matches = re.findall(region_pattern, full_text)
        region = region_matches[0] if region_matches else ""
        
        return {
            "region": region,
            "class_number": class_number,
            "hiragana": hiragana,
            "plate_number": plate_number,
            "raw": full_text
        }
    
    def format_plate_display(self, plate_info):
        """
        ナンバープレート情報を日本形式でフォーマット
        """
        region = plate_info.get("region", "") or "---"
        class_num = plate_info.get("class_number", "") or "---"
        hira = plate_info.get("hiragana", "") or "-"
        plate_num = plate_info.get("plate_number", "") or "----"
        
        # ハイフン形式に整形
        if len(plate_num) == 4 and "-" not in plate_num:
            plate_num = plate_num[:2] + "-" + plate_num[2:]
        
        return f"{region} {class_num}\n{hira} {plate_num}"

    def update(self, frame, plate_detections, track_id):
        """
        frame: BGR ndarray
        plate_detections:
        [
          { "bbox": (x1,y1,x2,y2) }
        ]
        """

        for det in plate_detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = frame[y1:y2, x1:x2]

            if crop is None or crop.size == 0:
                continue
            
            # 元のクロップを保存（表示用）
            self.plate_images[track_id] = crop.copy()
            
            # 前処理（必須：拡大→グレースケール→コントラスト強調→二値化）
            processed = self.preprocess_plate(crop)
            
            if processed is None:
                continue
            
            all_texts = []
            
            # EasyOCRで認識
            try:
                result = self.reader.readtext(processed, detail=0)
                if result:
                    all_texts.extend(result)
            except Exception as e:
                print(f"OCR error: {e}")
                continue
            
            if not all_texts:
                continue

            # 日本語ナンバープレート形式で解析
            plate_info = self.parse_japanese_plate(all_texts)
            self.results[track_id] = plate_info

    def get_plate_text(self, track_id):
        """フォーマット済みのナンバープレート文字列を返す"""
        plate_info = self.results.get(track_id)
        if plate_info is None:
            return "未検出"
        return self.format_plate_display(plate_info)
    
    def get_plate_info(self, track_id):
        """ナンバープレート情報の辞書を返す"""
        return self.results.get(track_id, {
            "region": "",
            "class_number": "",
            "hiragana": "",
            "plate_number": "",
            "raw": ""
        })
    
    def get_plate_image(self, track_id):
        """ナンバープレートのクロップ画像を返す"""
        return self.plate_images.get(track_id)

