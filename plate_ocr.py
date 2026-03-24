# plate_ocr.py
# ナンバープレートOCR（EasyOCR / Tesseract 切り替え版）

import os
import cv2
import numpy as np
from PIL import Image

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "EDSR_x2.pb")

# ============================================================
# OCRエンジン切り替え変数
#   "easyocr"   : EasyOCR (GPU対応、日本語精度高)
#   "tesseract" : Tesseract OCR (軽量、GPU不要)
# ============================================================
OCR_ENGINE = "tesseract"

_easyocr_available = False
_pytesseract_available = False

try:
    import easyocr as _easyocr_module
    _easyocr_available = True
except ImportError:
    pass

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    _pytesseract_available = True
except ImportError:
    pass


class PlateOCR:
    def __init__(self, engine=None):
        # エンジン選択（引数 > モジュール変数 > デフォルト）
        self.engine = (engine or OCR_ENGINE).lower()

        if self.engine == "easyocr":
            if not _easyocr_available:
                raise ImportError("easyocr が必要です。pip install easyocr")
            print("初期化中: EasyOCR...")
            self.reader = _easyocr_module.Reader(['ja', 'en'], gpu=True)
            self._tess_config_full = None
            self._tess_config_line = None
            self._tess_config_word = None
            print("EasyOCR初期化完了")
        else:
            if not _pytesseract_available:
                raise ImportError("pytesseract が必要です。pip install pytesseract")
            print("初期化中: Tesseract OCR...")
            self.reader = None
            # OEM 1=LSTMエンジン / PSM 6=ブロックテキスト / PSM 7=1行 / PSM 8=1単語
            self._tess_config_full = '--oem 1 --psm 6 -l jpn+eng'
            self._tess_config_line = '--oem 1 --psm 7 -l jpn+eng'
            self._tess_config_word = '--oem 1 --psm 8 -l jpn+eng'
            print("Tesseract OCR初期化完了")
        
        self.results = {}           # track_id -> plate_text
        self.plate_images = {}      # track_id -> ナンバープレートのクロップ画像
        self.plate_bboxes = {}      # track_id -> 検出されたプレートのBBox
        
        # 日本のナンバープレート最大文字数（特殊車両を除く）
        self.max_plate_chars = 11
        
        # 許可文字リスト（ナンバープレートに出現する文字のみ、後処理フィルタ用）
        self.plate_allowlist = (
            '0123456789・'
            'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
            '札幌函館旭川室蘭釧路帯広北見夕張岩見沢苫小牧稚内美唄芦別江別赤平紋別士別名寄三笠根室千歳滝川砂川歌志内深川富良野登別恵庭伊達北広島石狩北斗'
            '青森弘前八戸黒石五所川原十和田三沢むつつがる平川'
            '盛岡宮古大船渡花巻北上久慈遠野一関陸前高田釜石二戸八幡平奥州滝沢'
            '仙台石巻塩竈気仙沼白石名取角田多賀城岩沼登米栗原東松島大崎富谷'
            '秋田能代横手大館男鹿湯沢鹿角由利本荘潟上大仙北秋田にかほ仙北'
            '山形米沢鶴岡酒田新庄寒河江上山村山長井天童東根尾花沢南陽'
            '福島会津郡山いわき白河須賀川喜多方相馬二本松田村南相馬伊達本宮'
            '水戸日立土浦古河石岡結城龍ケ崎下妻常総常陸太田高萩北茨城笠間取手牛久つくば'
            '宇都宮足利栃木佐野鹿沼日光小山真岡大田原矢板那須塩原さくら那須烏山下野'
            '前橋高崎桐生伊勢崎太田沼田館林渋川藤岡富岡安中みどり'
            '大宮川口川越所沢熊谷春日部越谷'
            '千葉成田習志野柏松戸船橋市川木更津袖ケ浦野田'
            '品川練馬足立多摩八王子世田谷杉並'
            '横浜川崎相模湘南'
            '新潟長岡上越'
            '富山高岡'
            '金沢小松'
            '福井'
            '山梨甲府'
            '長野松本諏訪'
            '岐阜飛騨'
            '静岡浜松沼津伊豆'
            '名古屋豊橋岡崎豊田三河尾張小牧一宮春日井'
            '三重四日市鈴鹿'
            '滋賀大津'
            '京都'
            '大阪なにわ和泉堺'
            '神戸姫路'
            '奈良'
            '和歌山'
            '鳥取'
            '島根松江出雲'
            '岡山倉敷'
            '広島福山'
            '山口下関'
            '徳島'
            '香川高松'
            '愛媛松山'
            '高知'
            '福岡久留米北九州筑豊'
            '佐賀'
            '長崎佐世保'
            '熊本'
            '大分'
            '宮崎'
            '鹿児島'
            '沖縄那覇'
        )
    
    def _auto_white_balance(self, image):
        """簡易ホワイトバランス補正（BGR画像用）"""
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _adaptive_binarize(self, gray, level, thresh=120):

        """グレースケールになっていない場合はここで変換する"""
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        """複数の二値化手法を試し、文字領域が最も鮮明なものを返す"""
        candidates = []

        # 閾値
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        candidates.append(result)
        
        # # Otsu
        # _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # candidates.append(otsu)
        
        # # 適応的二値化 (Gaussian)
        # block = max(11, (gray.shape[1] // 8) | 1)  # 奇数にする
        # adapt_gauss = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 4 + level
        # )
        # candidates.append(adapt_gauss)
        
        # # 適応的二値化 (Mean)
        # adapt_mean = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, 6 + level
        # )
        # candidates.append(adapt_mean)
        
        # # 各候補のコントラスト（白と黒の分離度）で最適なものを選択
        # best = candidates[0]
        # best_score = 0
        # for c in candidates:
        #     white_ratio = np.sum(c == 255) / c.size
        #     # 白:黒が 30-70% の範囲内に近いほど良い
        #     score = 1.0 - abs(white_ratio - 0.5) * 2
        #     if score > best_score:
        #         best_score = score
        #         best = c
        return result

    def enhance_plate_image(self, image, level=1, pts=None, dst_size=(440, 220),
                            use_grayscale=True, use_denoise=True, use_superres=True,
                            use_normalize=True, use_binarize=True, binarize_thresh=120):
        """ナンバープレート画像をすべての補正を同時に段階的強度で適用
        
        全処理を常に実行し、レベルに応じて強度を段階的に調整：
          [透視変換] → グレースケール → 二値化
        
        Args:
            image:         入力画像 (BGR)
            level:         補正レベル (0=無補正, 1-5)
            pts:           透視変換用4点座標リスト [(x,y)...] (TL→TR→BR→BL 順)
                           指定時は最初に透視変換を実行する
            dst_size:      透視変換後の出力サイズ (w, h)
            use_grayscale: グレースケール変換を行うか
            use_denoise:   ノイズ除去を行うか
            use_superres:  超解像リサイズを行うか
            use_normalize: 正規化（コントラスト調整）を行うか
            use_binarize:  二値化を行うか
            binarize_thresh: 二値化の閾値 (0-255)
        Returns:
            補正後の画像 (BGR)
        """
        if image is None or image.size == 0:
            return image

        enhanced = image.copy()
        dst_w, dst_h = dst_size

        # === 透視変換 ===
        # pts が指定されていない場合は画像の4頂点をそのまま使用（dst_size へリサイズ）
        if pts is None:
            h, w = enhanced.shape[:2]
            pts = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        pts_arr = np.array(pts, dtype=np.float32)
        dst_arr = np.array(
            [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
            dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts_arr, dst_arr)
        enhanced = cv2.warpPerspective(enhanced, M, (dst_w, dst_h))

        # level=0 は透視変換のみで返す
        if level == 0:
            return enhanced

        # レベルに応じたパラメータ
        params = {
            1: {'scale': 3, 'denoise_h': 5,  'clahe': 1.5, 'gamma': 1.05, 'unsharp_sigma': 1.5, 'unsharp_w': 1.15, 'lap_w': 0.08, 'morph_k': 2, 'morph_i': 1},
            2: {'scale': 3, 'denoise_h': 7,  'clahe': 2.0, 'gamma': 1.10, 'unsharp_sigma': 2.0, 'unsharp_w': 1.25, 'lap_w': 0.15, 'morph_k': 2, 'morph_i': 1},
            3: {'scale': 4, 'denoise_h': 10, 'clahe': 2.5, 'gamma': 1.15, 'unsharp_sigma': 2.0, 'unsharp_w': 1.40, 'lap_w': 0.20, 'morph_k': 3, 'morph_i': 1},
            4: {'scale': 4, 'denoise_h': 12, 'clahe': 3.5, 'gamma': 1.25, 'unsharp_sigma': 2.5, 'unsharp_w': 1.60, 'lap_w': 0.30, 'morph_k': 3, 'morph_i': 1},
            5: {'scale': 5, 'denoise_h': 15, 'clahe': 4.5, 'gamma': 1.35, 'unsharp_sigma': 3.0, 'unsharp_w': 1.80, 'lap_w': 0.40, 'morph_k': 3, 'morph_i': 2},
        }
        p = params.get(level, params[3])

        # グレースケール変換 #
        if use_grayscale and len(enhanced.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # ノイズ除去 #
        if use_denoise:
            enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)

        # 正規化（コントラスト調整） — uint8 を保証 #
        if use_normalize:
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            if enhanced.dtype != np.uint8:
                enhanced = enhanced.astype(np.uint8)
        
        # 超解像リサイズ #
        if use_superres:
            h, w = enhanced.shape[:2]
            scale = p['scale']

            # グレースケールになっている場合はBGR変換する
            if len(enhanced.shape) == 2:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

            if os.path.isfile(_MODEL_PATH):
                # EDSR 超解像
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(_MODEL_PATH)
                sr.setModel("edsr", 2)
                enhanced = sr.upsample(enhanced)
            else:
                # モデルファイルが存在しない場合は cv2.resize にフォールバック
                interp = cv2.INTER_LANCZOS4 if level >= 3 else cv2.INTER_CUBIC
                enhanced = cv2.resize(enhanced, (w * scale, h * scale), interpolation=interp)

        # 二値化 #
        if use_binarize:
            enhanced = self._adaptive_binarize(enhanced, level, thresh=binarize_thresh)
        
        # BGR変換（表示・OCR共通） — 必ず 3ch BGR で返す
        if enhanced is None or enhanced.size == 0 or len(enhanced.shape) < 2:
            # フォールバック: 透視変換だけやり直して返す
            enhanced = cv2.warpPerspective(image, M, (dst_w, dst_h))
        elif len(enhanced.shape) == 2:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # デバッグ用
        print("RETURN TYPE:", type(enhanced))
        print("SHAPE:", getattr(enhanced, "shape", None))
        print("NDIM:", getattr(enhanced, "ndim", None))

        return enhanced
    
    # ===== 日本ナンバープレート地域名 =====
    _AREA_NAMES = {
        '札幌','函館','旭川','室蘭','釧路','帯広','北見','夕張','岩見沢','苫小牧',
        '稚内','美唄','芦別','江別','赤平','紋別','士別','名寄','三笠','根室',
        '千歳','滝川','砂川','歌志内','深川','富良野','登別','恵庭','伊達','北広島',
        '石狩','北斗','青森','弘前','八戸','黒石','五所川原','十和田','三沢',
        'むつ','つがる','平川','盛岡','宮古','大船渡','花巻','北上','久慈','遠野',
        '一関','陸前高田','釜石','二戸','八幡平','奥州','滝沢','仙台','石巻',
        '塩竈','気仙沼','白石','名取','角田','多賀城','岩沼','登米','栗原',
        '東松島','大崎','富谷','秋田','能代','横手','大館','男鹿','湯沢','鹿角',
        '由利本荘','潟上','大仙','北秋田','にかほ','仙北','山形','米沢','鶴岡',
        '酒田','新庄','寒河江','上山','村山','長井','天童','東根','尾花沢','南陽',
        '福島','会津','郡山','いわき','白河','須賀川','喜多方','相馬','二本松',
        '田村','南相馬','伊達','本宮','水戸','日立','土浦','古河','石岡','結城',
        '龍ケ崎','下妻','常総','常陸太田','高萩','北茨城','笠間','取手','牛久',
        'つくば','宇都宮','足利','栃木','佐野','鹿沼','日光','小山','真岡',
        '大田原','矢板','那須塩原','さくら','那須烏山','下野','前橋','高崎',
        '桐生','伊勢崎','太田','沼田','館林','渋川','藤岡','富岡','安中','みどり',
        '大宮','川口','川越','所沢','熊谷','春日部','越谷','千葉','成田',
        '習志野','柏','松戸','船橋','市川','木更津','袖ケ浦','野田','品川',
        '練馬','足立','多摩','八王子','世田谷','杉並','横浜','川崎','相模',
        '湘南','新潟','長岡','上越','富山','高岡','金沢','小松','福井','山梨',
        '甲府','長野','松本','諏訪','岐阜','飛騨','静岡','浜松','沼津','伊豆',
        '名古屋','豊橋','岡崎','豊田','三河','尾張','小牧','一宮','春日井',
        '三重','四日市','鈴鹿','滋賀','大津','京都','大阪','なにわ','和泉',
        '堺','神戸','姫路','奈良','和歌山','鳥取','島根','松江','出雲','岡山',
        '倉敷','広島','福山','山口','下関','徳島','香川','高松','愛媛','松山',
        '高知','福岡','久留米','北九州','筑豊','佐賀','長崎','佐世保','熊本',
        '大分','宮崎','鹿児島','沖縄','那覇',
    }

    # ===== ナンバープレートのひらがな（事業用・自家用） =====
    _PLATE_HIRAGANA = set(
        'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ'
        'まみむめもやゆよらりるれろわをん'
    )

    # 下段左（ひらがな1文字）専用許可文字
    _HIRAGANA_ALLOWLIST = (
        'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ'
        'まみむめもやゆよらりるれろわをん'
    )

    # 下段右（一連指定番号）専用許可文字
    _SERIAL_ALLOWLIST = '0123456789-'

    def _parse_plate(self, raw_text):
        """生のOCRテキストを日本のナンバープレート形式にパースする
        
        期待フォーマット:
          上段: 地名 3桁番号
          下段: ひらがな1文字 2桁-2桁 (数字は・に置換される場合あり)
        
        Returns:
            dict with keys: area, class_num, kana, reg_left, reg_right, formatted, score
            パース失敗時は None
        """
        import re
        if not raw_text:
            return None
        
        # 全角→半角, 不要文字除去
        text = raw_text.replace('\u3000', ' ')
        # ・（中黒）はそのまま保持
        # O/o → 0, I/l → 1 の典型OCR誤認識を補正
        text = text.replace('O', '0').replace('o', '0')
        text = text.replace('I', '1').replace('l', '1')
        
        # まず生テキストから各構成要素を抽出
        # 漢字部分 → 地域名候補
        kanji_parts = re.findall(r'[\u4e00-\u9fff\u3005\u3400-\u4dbf]+', text)
        # ひらがな
        hira_parts = re.findall(r'[\u3041-\u3096]', text)
        # 数字列 (・も数字的文字として扱う)
        digit_parts = re.findall(r'[0-9・·.]+', text)
        
        # --- 地域名の検出 ---
        area = ""
        for kp in kanji_parts:
            # 完全一致または部分一致で地域名を探す
            for a_name in self._AREA_NAMES:
                if a_name in kp or kp in a_name:
                    area = a_name
                    break
            if area:
                break
        # 地域名が見つからない場合、漢字部分の最初をそのまま使用
        if not area and kanji_parts:
            area = kanji_parts[0]
        
        # --- 分類番号 (3桁) ---
        class_num = ""
        pattern = re.compile(r'[0-9][0-9PXY][0-9ACFHKLMPXY]')
        for dp in digit_parts:
            cleaned = dp.replace('・', '').replace('·', '').replace('.', '')
            if pattern.fullmatch(cleaned):
                class_num = cleaned
                break
        # 3桁が見つからない場合、1-3桁の最初の数字列を使用
        if not class_num:
            for dp in digit_parts:
                cleaned = dp.replace('・', '').replace('·', '').replace('.', '')
                if 1 <= len(cleaned) <= 3 and cleaned.isdigit():
                    class_num = cleaned
                    break
        
        # --- ひらがな (1文字) ---
        kana = ""
        for h in hira_parts:
            if h in self._PLATE_HIRAGANA:
                kana = h
                break
        if not kana and hira_parts:
            kana = hira_parts[0]
        
        # --- 指定番号 (XX-XX, ・を含む場合あり) ---
        reg_left = ""
        reg_right = ""
        # 分類番号を除いた残りの数字/・列から指定番号を復元
        remaining_digits = []
        used_class = False
        for dp in digit_parts:
            cleaned = dp.replace('・', '').replace('·', '').replace('.', '')
            if cleaned == class_num and not used_class:
                used_class = True
                continue
            remaining_digits.append(dp)
        
        if remaining_digits:
            # ハイフンで区切られている場合
            joined = ''.join(remaining_digits)
            # ・や.を保持したまま処理
            # 4文字分あれば2-2に分割
            norm = joined.replace('·', '・').replace('.', '・')
            # ハイフンがある場合
            if '-' in norm:
                parts = norm.split('-', 1)
                reg_left = parts[0][-2:] if len(parts[0]) >= 2 else parts[0]
                reg_right = parts[1][:2] if len(parts[1]) >= 2 else parts[1]
            else:
                # 連続の場合、末尾4文字を2-2に
                chars = list(norm.replace('-', ''))
                if len(chars) >= 4:
                    reg_left = ''.join(chars[-4:-2])
                    reg_right = ''.join(chars[-2:])
                elif len(chars) >= 2:
                    mid = len(chars) // 2
                    reg_left = ''.join(chars[:mid])
                    reg_right = ''.join(chars[mid:])
        
        # --- スコアリング ---
        score = 0
        if area in self._AREA_NAMES:
            score += 3  # 正式な地域名
        elif area:
            score += 1
        if class_num and len(class_num) == 3:
            score += 2
        elif class_num:
            score += 1
        if kana and kana in self._PLATE_HIRAGANA:
            score += 2
        elif kana:
            score += 1
        # 指定番号は各桁が数字か・であること
        for c in (reg_left + reg_right):
            if c.isdigit() or c == '・':
                score += 0.5
        
        # 最低限の構成要素がない場合は失敗
        if score < 2:
            return None
        
        # --- フォーマット ---
        upper = f"{area} {class_num}" if area and class_num else (area or class_num or "")
        lower_reg = f"{reg_left}-{reg_right}" if reg_left or reg_right else ""
        lower = f"{kana} {lower_reg}" if kana else lower_reg
        formatted = f"{upper}\n{lower}" if upper and lower else (upper or lower or raw_text)
        
        return {
            'area': area,
            'class_num': class_num,
            'kana': kana,
            'reg_left': reg_left,
            'reg_right': reg_right,
            'formatted': formatted,
            'score': score,
        }

    def format_japanese_plate(self, text):
        """日本のナンバープレート形式に整形
        
        形式:
          地名 3桁
          ひらがな XX-XX
        
        改行が含まれる場合は上段・下段それぞれでパースを試み、
        最終的に2行形式で返す。
        """
        if not text:
            return "未検出"
        
        # 改行区切りがある場合、まず全体でパースを試みる
        joined = text.replace('\n', ' ')
        result = self._parse_plate(joined)
        if result:
            return result['formatted']
        
        # 全体パースが失敗し改行がある場合、各行をそのまま保持
        if '\n' in text:
            return text
        
        # パース失敗時はそのまま
        return text if text else "未検出"

    def count_recognized_chars(self, text):
        """認識された文字数をカウント（空白・記号・改行を除く）"""
        if not text:
            return 0
        import re
        cleaned = text.replace('\n', '')
        cleaned = re.sub(r'[\s\-\.:,、。]', '', cleaned)
        return len(cleaned)

    def _score_plate_text(self, text):
        """ナンバープレートの構造に基づくスコア（文字数だけでなく構造も評価）"""
        result = self._parse_plate(text)
        if result:
            return result['score']
        return self.count_recognized_chars(text) * 0.3  # 構造不一致は低スコア

    def _post_filter(self, text):
        """OCR生テキストからナンバープレート許可文字以外を除去する後処理フィルタ (案3)

        allowlistをEasyOCRに渡すとビームサーチが歪む場合があるため、
        制約なしOCRの結果を後処理でフィルタリングする方式。
        """
        if not text:
            return text
        allowed = set(self.plate_allowlist) | {' ', '\n', '-'}
        return ''.join(c for c in text if c in allowed)

    def _run_ocr(self, image, use_allowlist=True):
        """テキストを認識し返す（エンジンに応じて切り替え）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        try:
            if self.engine == "easyocr":
                allowlist = self.plate_allowlist if use_allowlist else None
                results = self.reader.readtext(gray, allowlist=allowlist, detail=1)
                return ' '.join(text for _, text, _ in results).strip()
            else:
                pil_img = Image.fromarray(gray)
                text = pytesseract.image_to_string(pil_img, config=self._tess_config_full).strip()
            return self._post_filter(text) if use_allowlist else text
        except Exception:
            return ""

    def _run_ocr_with_boxes(self, image, use_allowlist=True):
        """バウンディングボックス付きOCRで上段/下段を分離してテキストを返す（エンジン切り替え対応）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        h = gray.shape[0]
        mid_y = h / 3
        upper_texts = []
        lower_texts = []

        try:
            if self.engine == "easyocr":
                allowlist = self.plate_allowlist if use_allowlist else None
                results = self.reader.readtext(gray, allowlist=allowlist, detail=1)
                for bbox, text, _ in results:
                    if not text.strip():
                        continue
                    center_y = (bbox[0][1] + bbox[2][1]) / 2
                    center_x = (bbox[0][0] + bbox[2][0]) / 2
                    if center_y < mid_y:
                        upper_texts.append((center_x, text.strip()))
                    else:
                        lower_texts.append((center_x, text.strip()))
            else:
                pil_img = Image.fromarray(gray)
                data = pytesseract.image_to_data(
                    pil_img, config=self._tess_config_full,
                    output_type=pytesseract.Output.DICT
                )
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    if not text:
                        continue
                    center_y = data['top'][i] + data['height'][i] / 2
                    center_x = data['left'][i] + data['width'][i] / 2
                    if center_y < mid_y:
                        upper_texts.append((center_x, text))
                    else:
                        lower_texts.append((center_x, text))
        except Exception:
            return ""

        upper_texts.sort(key=lambda x: x[0])
        lower_texts.sort(key=lambda x: x[0])

        upper = ' '.join(t for _, t in upper_texts)
        lower = ' '.join(t for _, t in lower_texts)

        result = f"{upper} {lower}" if upper or lower else ""
        if self.engine != "easyocr" and use_allowlist:
            result = self._post_filter(result)
        return result

    def _run_ocr_split(self, image, use_allowlist=True):
        """画像を上1:下2の高さ比で物理的に分割してOCR

        ナンバープレートの上段(地域名+分類番号)と下段(ひらがな+登録番号)を
        別々にクロップしてOCRすることで精度を向上させる。

        下段はさらに左右に分割し、領域別のTesseract設定を適用：
          左部分（幅の約20%）: ひらがな1文字想定
          右部分（幅の約80%）: 数字・ハイフン想定

        Returns:
            str: "上段テキスト\n下段テキスト" 形式
        """
        h, w = image.shape[:2]
        if h < 6 or w < 6:
            return ""

        # 上段: 高さの1/3
        split_y = h // 3
        upper_crop = image[0:split_y, :]
        # 下段: 高さの2/3
        lower_crop = image[split_y:, :]

        # 下段の左右分割（左約20%: ひらがな、右約80%: 一連指定番号）
        kana_split_x = max(1, w // 5)
        lower_kana_crop   = lower_crop[:, :kana_split_x]
        lower_serial_crop = lower_crop[:, kana_split_x:]

        def _tess(crop, config):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
            pil_img = Image.fromarray(gray)
            try:
                return pytesseract.image_to_string(pil_img, config=config).strip()
            except Exception:
                return ""

        def _easy(crop, allowlist=None):
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()
            try:
                results = self.reader.readtext(gray, allowlist=allowlist, detail=1)
                return ' '.join(text for _, text, _ in results).strip()
            except Exception:
                return ""

        if self.engine == "easyocr":
            al = self.plate_allowlist if use_allowlist else None
            upper_text  = _easy(upper_crop,        al)
            kana_text   = _easy(lower_kana_crop,   self._HIRAGANA_ALLOWLIST if use_allowlist else None)
            serial_text = _easy(lower_serial_crop, self._SERIAL_ALLOWLIST if use_allowlist else None)
        else:
            upper_text  = _tess(upper_crop,        self._tess_config_line)
            kana_text   = _tess(lower_kana_crop,   self._tess_config_word)
            serial_text = _tess(lower_serial_crop, self._tess_config_line)
            if use_allowlist:
                al_full   = set(self.plate_allowlist) | {' ', '\n', '-'}
                al_kana   = set(self._HIRAGANA_ALLOWLIST) | {' '}
                al_serial = set(self._SERIAL_ALLOWLIST) | {' ', '-'}
                upper_text  = ''.join(c for c in upper_text  if c in al_full)
                kana_text   = ''.join(c for c in kana_text   if c in al_kana)
                serial_text = ''.join(c for c in serial_text if c in al_serial)

        lower_text = f"{kana_text} {serial_text}".strip()

        if upper_text or lower_text:
            return f"{upper_text}\n{lower_text}"
        return ""

    def _resize_to_original(self, image, orig_size):
        """画像を元のクロップサイズにリサイズ（OCR用）"""
        if orig_size is None:
            return image
        oh, ow = orig_size
        h, w = image.shape[:2]
        if h == oh and w == ow:
            return image
        return cv2.resize(image, (ow, oh), interpolation=cv2.INTER_AREA)

    def _run_all_ocr_patterns(self, image, inv=None, orig_size=None):
        """1つの画像に対して全OCRパターンを実行し候補リストを返す
        
        orig_sizeが指定されている場合、OCR実行前にそのサイズにリサイズしてからOCRにかける。
        表示用画像は拡大済みのまま、OCR用は元のクロップサイズ。
        """
        ocr_img = self._resize_to_original(image, orig_size)
        candidates = []
        candidates.append(self._run_ocr(ocr_img, use_allowlist=True))
        candidates.append(self._run_ocr_with_boxes(ocr_img, use_allowlist=True))
        candidates.append(self._run_ocr_split(ocr_img, use_allowlist=True))
        # allowlistなし + 後処理フィルタ (案3)
     #   raw_no_al = self._run_ocr(ocr_img, use_allowlist=False)
     #   candidates.append(raw_no_al)
     #   candidates.append(self._post_filter(raw_no_al))

        if inv is not None:
            ocr_inv = self._resize_to_original(inv, orig_size)
            candidates.append(self._run_ocr(ocr_inv, use_allowlist=True))
            candidates.append(self._run_ocr_with_boxes(ocr_inv, use_allowlist=True))
            candidates.append(self._run_ocr_split(ocr_inv, use_allowlist=True))
            raw_inv_no_al = self._run_ocr(ocr_inv, use_allowlist=False)
            candidates.append(raw_inv_no_al)
            candidates.append(self._post_filter(raw_inv_no_al))

        return candidates

    def _pick_best_candidate(self, candidates):
        """候補リストから構造スコア最良のテキストとスコアを返す"""
        best_text = ""
        best_score = 0
        for ct in candidates:
            s = self._score_plate_text(ct)
            if s > best_score:
                best_score = s
                best_text = ct
        return best_text, best_score

    def _vote_candidates(self, candidates):
        """複数OCR候補から構成要素を多数決で統合する (案2)

        全候補をパースし、地名・分類番号・かな・指定番号それぞれについて
        最頻値で最終値を決定する。単純な最高スコア選択より1フレーム内の
        OCRパターン間での誤認識に強い。
        パース可能な候補が2つ未満の場合は_pick_best_candidateにフォールバック。
        """
        from collections import Counter

        parsed_results = []
        for ct in candidates:
            if not ct:
                continue
            r = self._parse_plate(ct)
            if r and r['score'] >= 2:
                parsed_results.append(r)

        best_single, best_single_score = self._pick_best_candidate(candidates)

        if len(parsed_results) < 2:
            return best_single, best_single_score

        area_votes  = Counter(r['area']      for r in parsed_results if r['area'])
        class_votes = Counter(r['class_num'] for r in parsed_results if r['class_num'])
        kana_votes  = Counter(r['kana']      for r in parsed_results if r['kana'])
        regl_votes  = Counter(r['reg_left']  for r in parsed_results if r['reg_left'])
        regr_votes  = Counter(r['reg_right'] for r in parsed_results if r['reg_right'])

        area      = area_votes.most_common(1)[0][0]  if area_votes  else ""
        class_num = class_votes.most_common(1)[0][0] if class_votes else ""
        kana      = kana_votes.most_common(1)[0][0]  if kana_votes  else ""
        reg_left  = regl_votes.most_common(1)[0][0]  if regl_votes  else ""
        reg_right = regr_votes.most_common(1)[0][0]  if regr_votes  else ""

        upper     = f"{area} {class_num}".strip()
        lower_reg = f"{reg_left}-{reg_right}" if reg_left or reg_right else ""
        lower     = f"{kana} {lower_reg}".strip()
        voted_text = f"{upper}\n{lower}".strip()

        voted_score = 0
        if area in self._AREA_NAMES:
            voted_score += 3
        elif area:
            voted_score += 1
        if class_num and len(class_num) == 3:
            voted_score += 2
        elif class_num:
            voted_score += 1
        if kana and kana in self._PLATE_HIRAGANA:
            voted_score += 2
        elif kana:
            voted_score += 1
        for c in (reg_left + reg_right):
            if c.isdigit() or c == '・':
                voted_score += 0.5
        # 複数候補が一致するほど信頼度が高い（最大+1.0ボーナス）
        voted_score += min(1.0, len(parsed_results) / max(len(candidates), 1))

        return (voted_text, voted_score) if voted_score >= best_single_score else (best_single, best_single_score)

    def ocr_with_progressive_enhancement(self, plate_crop, callback=None):
        """段階的に全補正の強度を上げながらOCRを実行
        
        レベル0: 補正なし（オリジナル）
        レベル1-5: 段階的補正
        レベル6: 白黒反転のみ（表示更新なし）
        
        上1:下2で画像を物理分割するパターンも毎レベルで実行。
        
        Args:
            plate_crop: ナンバープレートのクロップ画像
            callback: (level, original_img, enhanced_img, text, char_count)
        Returns:
            tuple: (best_text, best_image)
        """
        import time
        
        best_text = ""
        best_image = plate_crop.copy()
        best_score = 0
        original_image = plate_crop.copy()  # オリジナル保持
        orig_size = (plate_crop.shape[0], plate_crop.shape[1])  # 元のクロップサイズ
        all_candidates = []  # 全レベルの候補を蓄積（案2: 多数決用）

        # ===== レベル0: 補正なし =====
        try:
            candidates = self._run_all_ocr_patterns(original_image)
            all_candidates.extend(candidates)
            level_best_text, level_best_score = self._vote_candidates(candidates)
            level_count = self.count_recognized_chars(level_best_text)
            display_text = self.format_japanese_plate(level_best_text)
            
            if callback:
                callback(0, original_image, original_image, display_text, level_count)
            
            if level_best_score > best_score:
                best_text = level_best_text
                best_image = original_image.copy()
                best_score = level_best_score
                print(f"レベル 0 (オリジナル): score={best_score:.1f} → {display_text!r}")
            
            # スコアが十分高ければ補正スキップ
            if best_score >= 8.0:
                print(f"OCR処理完了(補正不要): score={best_score:.1f} → {display_text}")
                return best_text, best_image
            
            time.sleep(0.8)
        except Exception as e:
            print(f"Level 0 (original) error: {e}")
        
        # ===== レベル1-5: 段階的補正 =====
        for level in range(1, 6):
            try:
                enhanced = self.enhance_plate_image(plate_crop, level=level)
                inv = cv2.bitwise_not(enhanced)
                
                candidates = self._run_all_ocr_patterns(enhanced, inv, orig_size=orig_size)
                all_candidates.extend(candidates)
                level_best_text, level_best_score = self._vote_candidates(candidates)
                level_count = self.count_recognized_chars(level_best_text)
                display_text = self.format_japanese_plate(level_best_text)
                
                if callback:
                    callback(level, original_image, enhanced, display_text, level_count)
                
                if level_best_score > best_score:
                    best_text = level_best_text
                    best_image = enhanced.copy()
                    best_score = level_best_score
                    print(f"レベル {level}: score={best_score:.1f} → {display_text!r}")
                else:
                    print(f"レベル {level}: 改善なし (score={level_best_score:.1f}, best={best_score:.1f})")
                
                time.sleep(0.8)
            except Exception as e:
                print(f"Enhancement level {level} error: {e}")
                continue
        
        # ===== レベル6: 白黒反転のみ（表示更新なし） =====
        try:
            inverted_original = cv2.bitwise_not(original_image)
            candidates = self._run_all_ocr_patterns(inverted_original, orig_size=None)
            all_candidates.extend(candidates)
            level_best_text, level_best_score = self._vote_candidates(candidates)

            if level_best_score > best_score:
                best_text = level_best_text
                best_image = inverted_original.copy()
                best_score = level_best_score
                display_text = self.format_japanese_plate(level_best_text)
                print(f"レベル 6 (反転): score={best_score:.1f} → {display_text!r}")
            # レベル6はコールバックしない（表示更新なし）
        except Exception as e:
            print(f"Level 6 (inverted) error: {e}")

        # ===== 最終: 全レベル候補による多数決 (案2) =====
        if all_candidates:
            final_text, final_score = self._vote_candidates(all_candidates)
            if final_score > best_score:
                best_text = final_text
                best_score = final_score
                print(f"最終多数決: score={best_score:.1f} → {self.format_japanese_plate(best_text)!r}")

        formatted = self.format_japanese_plate(best_text)
        print(f"OCR処理完了: score={best_score:.1f} → {formatted}")
        return best_text, best_image
    
    def update(self, frame, car_bbox, track_id, plate_bbox=None, use_progressive=False, callback=None):
        """
        ナンバープレートのクロップからOCR
        
        frame: BGR ndarray (元フレーム全体)
        car_bbox: (x1, y1, x2, y2) 車のBBox
        track_id: 追跡ID
        plate_bbox: (x1, y1, x2, y2) ナンバープレートのBBox（YOLOで検出済み）
        use_progressive: 段階的補正を使用するか
        callback: 段階的補正時のコールバック関数
        """
        # plate_bboxがある場合はそれを使用、なければ車全体を使用
        if plate_bbox is not None:
            px1, py1, px2, py2 = plate_bbox
            plate_crop = frame[py1:py2, px1:px2]
            
            if plate_crop is None or plate_crop.size == 0:
                # フォールバック：車のクロップを使用
                x1, y1, x2, y2 = car_bbox
                car_crop = frame[y1:y2, x1:x2]
                plate_crop = car_crop
        else:
            # plate_bboxがない場合は車のクロップを使用
            x1, y1, x2, y2 = car_bbox
            car_crop = frame[y1:y2, x1:x2]
            plate_crop = car_crop
        
        if plate_crop is None or plate_crop.size == 0:
            return
        
        # 段階的補正を使用する場合
        if use_progressive:
            best_text, best_image = self.ocr_with_progressive_enhancement(plate_crop, callback)
            self.results[track_id] = best_text
            self.plate_images[track_id] = best_image
            return
        
        # 通常のOCR（段階的補正なし）
        try:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
            if self.engine == "easyocr":
                results = self.reader.readtext(gray, detail=1)
                plate_text = ' '.join(text for _, text, _ in results).strip()
            else:
                pil_img = Image.fromarray(gray)
                plate_text = self._post_filter(
                    pytesseract.image_to_string(pil_img, config=self._tess_config_full).strip()
                )

            self.results[track_id] = plate_text
            self.plate_images[track_id] = plate_crop.copy()
                
        except Exception as e:
            print(f"OCR error for track_id {track_id}: {e}")
            import traceback
            traceback.print_exc()
            self.results[track_id] = ""
            self.plate_images[track_id] = plate_crop.copy()
    
    def get_plate_text(self, track_id):
        """ナンバープレート文字列を返す（整形済み）"""
        text = self.results.get(track_id, "")
        if not text:
            return "未検出"
        return self.format_japanese_plate(text)
    
    def get_plate_image(self, track_id):
        """ナンバープレートのクロップ画像を返す"""
        return self.plate_images.get(track_id)
