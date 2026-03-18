# plate_dialog.py
# ナンバープレート表示用ダイアログ（手動操作版）

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


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
        self._manual_pts = []           # 4点クリック [(x,y)...] オリジナル画像座標
        self._pts_status_label = None   # クリック状態ヒントラベル
        self._orig_display_scale = 1.0  # オリジナル画像の表示倍率
    
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

    # ================================================================
    # 4点クリック透視変換
    # ================================================================
    def _show_annotated_original(self):
        """クリック点と輪郭線を描画したオリジナル画像を左パネルに表示する"""
        vis = self.original_crop.copy()
        corner_labels = ["\u2460TL", "\u2461TR", "\u2462BR", "\u2463BL"]
        colors_bgr = [(0, 200, 0), (0, 200, 255), (255, 165, 0), (255, 0, 200)]
        for i, (px, py) in enumerate(self._manual_pts):
            ipt = (int(px), int(py))
            cv2.circle(vis, ipt, 1, colors_bgr[i], -1)
            cv2.circle(vis, ipt, 1, (255, 255, 255), 1)
            cv2.putText(vis, corner_labels[i], (ipt[0] + 3, ipt[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, colors_bgr[i], 1, cv2.LINE_AA)
        if len(self._manual_pts) >= 2:
            for i in range(len(self._manual_pts) - 1):
                cv2.line(vis,
                         (int(self._manual_pts[i][0]),     int(self._manual_pts[i][1])),
                         (int(self._manual_pts[i+1][0]),   int(self._manual_pts[i+1][1])),
                         (0, 200, 0), 1, cv2.LINE_AA)
        if len(self._manual_pts) == 4:
            cv2.line(vis,
                     (int(self._manual_pts[3][0]), int(self._manual_pts[3][1])),
                     (int(self._manual_pts[0][0]), int(self._manual_pts[0][1])),
                     (0, 200, 0), 1, cv2.LINE_AA)
        sc = self._orig_display_scale
        if sc > 1.0:
            vis = cv2.resize(vis,
                             (int(vis.shape[1] * sc), int(vis.shape[0] * sc)),
                             interpolation=cv2.INTER_NEAREST)
        photo = self._bgr_to_photo(vis)
        if photo and self.original_label:
            self.original_label.config(image=photo, width=photo.width(), height=photo.height())
            self.original_label.image = photo

    def _update_pts_hint(self):
        """クリック状態ヒントラベルを更新する"""
        if self._pts_status_label is None:
            return
        hints = [
            "\u2460 \u5de6\u4e0a(TL)\u3092\u30af\u30ea\u30c3\u30af",
            "\u2461 \u53f3\u4e0a(TR)\u3092\u30af\u30ea\u30c3\u30af",
            "\u2462 \u53f3\u4e0b(BR)\u3092\u30af\u30ea\u30c3\u30af",
            "\u2463 \u5de6\u4e0b(BL)\u3092\u30af\u30ea\u30c3\u30af",
            "\u2713 \u88dc\u6b63\u5b8c\u4e86  (\u30ea\u30bb\u30c3\u30c8\u3067\u518d\u8a66\u884c)",
        ]
        self._pts_status_label.config(text=hints[len(self._manual_pts)])

    def _on_orig_click(self, event):
        """オリジナル画像クリックで4点を収集し、4点揃ったら透視変換を適用する"""
        if self.is_closed or len(self._manual_pts) >= 4:
            return
        sc = self._orig_display_scale
        self._manual_pts.append((event.x / sc, event.y / sc))
        self._show_annotated_original()
        self._update_pts_hint()
        if len(self._manual_pts) == 4:
            self._apply_4pts_correction()

    def _apply_4pts_correction(self):
        """クリックされた4点から透視変換を適用し補正画像を生成・OCRを実行する"""
        if len(self._manual_pts) < 4:
            return
        pts = np.array(self._manual_pts, dtype=np.float32)
        # 出力サイズ: 日本ナンバープレート 330×165mm のアスペクト比 2:1
        dst_w, dst_h = 440, 220
        dst = np.array(
            [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
            dtype=np.float32)
        try:
            M = cv2.getPerspectiveTransform(pts, dst)
            corrected = cv2.warpPerspective(self.original_crop, M, (dst_w, dst_h))
        except Exception as e:
            print(f"Perspective transform error: {e}")
            return
        
        # 補正レベルプルダウンの選択値を使って 前処理(enhance_plate_image)を行う
        level_str = self.level_combo.get()
        if level_str != "オリジナル":
             level = int(level_str.replace("Lv.", ""))
             corrected = self.plate_ocr.enhance_plate_image(corrected, level=level)

        self.current_enhanced = corrected

        photo_enh = self._bgr_to_photo(corrected)
        if photo_enh and self.image_label:
            self.image_label.config(image=photo_enh,
                                    width=photo_enh.width(), height=photo_enh.height())
            self.image_label.image = photo_enh
        # 440×220 画像をそのままOCR (orig_size=None でリサイズしない)
        try:
            candidates = self.plate_ocr._run_all_ocr_patterns(corrected)
            best_text, best_score = self.plate_ocr._vote_candidates(candidates)
            display_text = self.plate_ocr.format_japanese_plate(best_text)
            char_count = self.plate_ocr.count_recognized_chars(best_text)
            if self.text_label:
                self.text_label.config(
                    text=display_text if display_text else "認識できませんでした")
            if self.progress_label:
                self.progress_label.config(
                    text=f"score: {best_score:.1f} | 認識文字数: {char_count}")
        except Exception as e:
            print(f"4点補正後OCRエラー: {e}")
            if self.text_label:
                self.text_label.config(text="OCRエラー")
        if self.root:
            self.root.update()

    def _reset_manual_pts(self):
        """クリックポイントをリセットし、元の表示に戻す"""
        self._manual_pts = []
        self._show_annotated_original()
        self._update_pts_hint()
        photo_raw = self._bgr_to_photo(self.original_crop)
        if photo_raw and self.image_label:
            self.image_label.config(image=photo_raw,
                                    width=photo_raw.width(), height=photo_raw.height())
            self.image_label.image = photo_raw
        self.current_enhanced = self.original_crop.copy()

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

        tk.Button(control_frame, text="4点リセット",
                  command=self._reset_manual_pts,
                  font=("Arial", 9), bg="#555555", fg="#ffffff",
                  activebackground="#666666", padx=6).pack(side=tk.LEFT, padx=(10, 0))
        
        # 進捗表示
        self.progress_label = tk.Label(
            main_frame,
            text="OCR実行中...",
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
        
        orig_title = tk.Label(orig_frame, text="オリジナル  (4点クリックで補正)",
                              font=("Arial", 9, "bold"), fg="#cccccc", bg="#2b2b2b")
        orig_title.pack()

        self._pts_status_label = tk.Label(orig_frame, text="\u2460 左上(TL)をクリック",
                                          font=("Arial", 8), fg="#ffcc00", bg="#2b2b2b")
        self._pts_status_label.pack()

        orig_border = tk.Frame(orig_frame, bg="#555555", bd=0)
        orig_border.pack()

        self.original_label = tk.Label(orig_border, bg="#1a1a1a")
        self.original_label.pack(padx=2, pady=2)
        self.original_label.bind("<Button-1>", self._on_orig_click)
        self.original_label.config(cursor="crosshair")

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
        
        # 表示スケールを計算（幅が300px未満なら拡大してクリックしやすくする）
        ih, iw = self.original_crop.shape[:2]
        self._orig_display_scale = max(1.0, 300.0 / max(iw, 1))

        # 初期表示：左はスケールアップ（クリックしやすくするため）、右はオリジナルサイズ
        if self._orig_display_scale > 1.0:
            disp_w = int(iw * self._orig_display_scale)
            disp_h = int(ih * self._orig_display_scale)
            disp_orig = cv2.resize(self.original_crop, (disp_w, disp_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            disp_orig = self.original_crop
        photo_disp = self._bgr_to_photo(disp_orig)
        photo_raw  = self._bgr_to_photo(self.original_crop)
        if photo_disp:
            self.original_label.config(image=photo_disp,
                                       width=photo_disp.width(), height=photo_disp.height())
            self.original_label.image = photo_disp
        if photo_raw:
            self.image_label.config(image=photo_raw,
                                    width=photo_raw.width(), height=photo_raw.height())
            self.image_label.image = photo_raw

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
            text="OCR実行中...",
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
        
        # オリジナル画像でOCRを自動実行
        self.root.after(100, self._on_level_changed)
    
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
