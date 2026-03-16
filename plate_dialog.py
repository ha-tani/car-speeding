# plate_dialog.py
# ナンバープレート表示用ダイアログ（手動操作版）

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


class PlateDialog:
    def __init__(self, track_id, plate_ocr, original_crop, plate_text=""):
        self.track_id = track_id
        self.plate_ocr = plate_ocr  # PlateOCRインスタンス
        self.original_crop = original_crop.copy()  # 元のナンバープレート画像
        self.plate_text = plate_text
        self.root = None
        self.hidden_root = None
        self.original_label = None  # 左: オリジナル画像
        self.image_label = None     # 右: 補正画像
        self.text_label = None
        self.progress_label = None
        self.level_combo = None     # 補正レベルプルダウン
        self.invert_var = None      # 白黒反転チェックボックス
        self.is_closed = False
        self.current_enhanced = None  # 現在の補正画像
    
    def _bgr_to_photo(self, image):
        """BGR画像をそのままのサイズでPhotoImageに変換"""
        if image is None or image.size == 0:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil_img)

    def _on_level_changed(self, event=None):
        """補正レベルまたは反転が変更された時の処理"""
        if self.is_closed or self.root is None:
            return
        
        try:
            # 選択されたレベルを取得
            level_str = self.level_combo.get()
            if level_str == "オリジナル":
                level = 0
            else:
                level = int(level_str.replace("Lv.", ""))
            
            # 白黒反転フラグ
            invert = self.invert_var.get()
            
            # 補正画像を生成
            if level == 0:
                enhanced = self.original_crop.copy()
            else:
                enhanced = self.plate_ocr.enhance_plate_image(self.original_crop, level=level)
            
            # 白黒反転
            if invert:
                enhanced = cv2.bitwise_not(enhanced)
            
            self.current_enhanced = enhanced
            
            # 補正画像を表示更新
            photo_enh = self._bgr_to_photo(enhanced)
            if photo_enh:
                self.image_label.config(image=photo_enh, width=photo_enh.width(), height=photo_enh.height())
                self.image_label.image = photo_enh
            
            # OCR実行
            self._run_ocr(enhanced)
            
            self.root.update()
        except Exception as e:
            print(f"Level change error: {e}")
            import traceback
            traceback.print_exc()

    def _run_ocr(self, image):
        """指定された画像にOCRを実行し結果を表示"""
        try:
            # 元のクロップサイズにリサイズ（OCR用）
            orig_size = (self.original_crop.shape[0], self.original_crop.shape[1])
            ocr_img = self.plate_ocr._resize_to_original(image, orig_size)
            
            # 全OCRパターンを実行し多数決で最良候補を選択
            candidates = self.plate_ocr._run_all_ocr_patterns(ocr_img)
            best_text, best_score = self.plate_ocr._vote_candidates(candidates)
            
            # テキストを整形して表示
            display_text = self.plate_ocr.format_japanese_plate(best_text)
            char_count = self.plate_ocr.count_recognized_chars(best_text)
            
            if display_text:
                self.text_label.config(text=display_text)
            else:
                self.text_label.config(text="認識できませんでした")
            
            self.progress_label.config(text=f"score: {best_score:.1f} | 認識文字数: {char_count}")
            
        except Exception as e:
            print(f"OCR error: {e}")
            import traceback
            traceback.print_exc()
            self.text_label.config(text="OCRエラー")
            self.progress_label.config(text="処理失敗")

    def show(self):
        """ダイアログを表示"""
        if self.root is not None:
            return
        
        # 非表示のルートウィンドウを作成（アイコン表示防止）
        self.hidden_root = tk.Tk()
        self.hidden_root.withdraw()
        
        # Toplevelダイアログを作成
        self.root = tk.Toplevel(self.hidden_root)
        self.root.title("ナンバープレート情報")
        self.root.configure(bg="#2b2b2b")
        
        # ウィンドウサイズは画像に合わせて自動調整
        self.root.resizable(True, True)
        
        # 閉じられた時のフラグ設定
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # メインフレーム
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # ===== 補正コントロールフレーム =====
        control_frame = tk.Frame(main_frame, bg="#2b2b2b")
        control_frame.pack(fill=tk.X, pady=(0, 8))
        
        # 補正レベルプルダウン
        tk.Label(control_frame, text="補正:", font=("Arial", 9),
                 fg="#cccccc", bg="#2b2b2b").pack(side=tk.LEFT, padx=(0, 5))
        
        self.level_combo = ttk.Combobox(control_frame, 
                                         values=["オリジナル", "Lv.1", "Lv.2", "Lv.3", "Lv.4", "Lv.5"],
                                         state="readonly", width=10)
        self.level_combo.set("オリジナル")
        self.level_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.level_combo.bind("<<ComboboxSelected>>", self._on_level_changed)
        
        # 白黒反転チェックボックス
        self.invert_var = tk.BooleanVar(value=False)
        invert_check = tk.Checkbutton(control_frame, text="白黒反転",
                                       variable=self.invert_var,
                                       command=self._on_level_changed,
                                       font=("Arial", 9), fg="#cccccc", bg="#2b2b2b",
                                       selectcolor="#1a1a1a", activebackground="#2b2b2b")
        invert_check.pack(side=tk.LEFT)
        
        # 進捗表示
        self.progress_label = tk.Label(
            main_frame,
            text="補正レベルを選択してOCRを実行",
            font=("Arial", 9),
            fg="#aaaaaa",
            bg="#2b2b2b"
        )
        self.progress_label.pack(pady=(0, 8))
        
        # ===== 画像表示フレーム =====
        image_container = tk.Frame(main_frame, bg="#2b2b2b")
        image_container.pack(pady=(0, 8))
        
        # 左: オリジナル画像
        orig_frame = tk.Frame(image_container, bg="#2b2b2b")
        orig_frame.pack(side=tk.LEFT, padx=(0, 5))
        
        orig_title = tk.Label(orig_frame, text="オリジナル", font=("Arial", 9, "bold"),
                              fg="#cccccc", bg="#2b2b2b")
        orig_title.pack()
        
        orig_border = tk.Frame(orig_frame, bg="#555555", bd=0)
        orig_border.pack()
        
        self.original_label = tk.Label(orig_border, bg="#1a1a1a")
        self.original_label.pack(padx=2, pady=2)
        
        # 右: 補正画像
        enh_frame = tk.Frame(image_container, bg="#2b2b2b")
        enh_frame.pack(side=tk.LEFT, padx=(5, 0))
        
        enh_title = tk.Label(enh_frame, text="補正画像", font=("Arial", 9, "bold"),
                            fg="#cccccc", bg="#2b2b2b")
        enh_title.pack()
        
        enh_border = tk.Frame(enh_frame, bg="#555555", bd=0)
        enh_border.pack()
        
        self.image_label = tk.Label(enh_border, bg="#1a1a1a")
        self.image_label.pack(padx=2, pady=2)
        
        # 初期表示：オリジナル画像を両方に表示
        photo_orig = self._bgr_to_photo(self.original_crop)
        if photo_orig:
            self.original_label.config(image=photo_orig, width=photo_orig.width(), height=photo_orig.height())
            self.original_label.image = photo_orig
            self.image_label.config(image=photo_orig, width=photo_orig.width(), height=photo_orig.height())
            self.image_label.image = photo_orig
        
        self.current_enhanced = self.original_crop.copy()
        
        # ===== OCR結果テキスト =====
        text_frame = tk.Frame(main_frame, bg="#3a3a3a", bd=1, relief=tk.SOLID)
        text_frame.pack(fill=tk.X, pady=(0, 5))
        
        text_title = tk.Label(
            text_frame,
            text="OCR結果",
            font=("Arial", 9, "bold"),
            fg="#cccccc",
            bg="#3a3a3a"
        )
        text_title.pack(anchor="w", padx=5, pady=(3, 0))
        
        self.text_label = tk.Label(
            text_frame,
            text="補正レベルを選択してください",
            font=("MS Gothic", 14, "bold"),
            fg="#00ff00",
            bg="#1a1a1a",
            justify=tk.LEFT,
            anchor="w"
        )
        self.text_label.pack(padx=8, pady=(3, 8), fill=tk.X)
        
        # ===== 閉じるボタン =====
        close_btn = tk.Button(
            main_frame,
            text="閉じる",
            command=self._on_closing,
            font=("Arial", 10),
            width=12,
            bg="#555555",
            fg="#ffffff",
            activebackground="#666666"
        )
        close_btn.pack(pady=(5, 0))
        
        # Escキーで閉じる
        self.root.bind("<Escape>", lambda e: self._on_closing())
        
        # ウィンドウを最前面に
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        # 初期描画を強制実行
        self.root.update_idletasks()
        self.root.update()
    
    def process_events(self):
        """Tkinterイベントを処理（メインループから定期的に呼び出す）"""
        if self.is_closed or self.root is None:
            return False
        try:
            self.root.update()
            return True
        except tk.TclError:
            self.is_closed = True
            self.root = None
            return False
    
    def _on_closing(self):
        """ダイアログを閉じる"""
        self.is_closed = True
        if self.root:
            self.root.destroy()
            self.root = None
        if self.hidden_root:
            self.hidden_root.destroy()
            self.hidden_root = None

    def wait_close(self):
        """ダイアログが閉じられるまで待機"""
        if self.root:
            self.root.wait_window()
