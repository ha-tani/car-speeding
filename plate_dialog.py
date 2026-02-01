# plate_dialog.py
# ナンバープレート表示用ダイアログ

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class PlateDialog:
    def __init__(self, track_id, plate_text, plate_image=None):
        self.track_id = track_id
        self.plate_text = plate_text
        self.plate_image = plate_image  # OpenCV BGR画像

    def show(self):
        root = tk.Tk()
        root.title("ナンバープレート情報")
        root.configure(bg="#2b2b2b")
        
        # ウィンドウサイズ
        root.geometry("350x300")
        root.resizable(False, False)

        # メインフレーム
        main_frame = tk.Frame(root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # ナンバープレート画像表示
        if self.plate_image is not None and self.plate_image.size > 0:
            # BGRからRGBに変換
            rgb_image = cv2.cvtColor(self.plate_image, cv2.COLOR_BGR2RGB)
            
            # 拡大表示（幅300pxに）
            h, w = rgb_image.shape[:2]
            if w > 0:
                scale = 280 / w
                new_w = 280
                new_h = int(h * scale)
                new_h = min(new_h, 120)  # 高さ制限
                
                rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # PIL Imageに変換
            pil_image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(pil_image)
            
            # 画像ラベル（枠付き）
            image_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief=tk.RAISED)
            image_frame.pack(pady=(0, 15))
            
            image_label = tk.Label(image_frame, image=photo, bg="#ffffff")
            image_label.image = photo  # 参照を保持
            image_label.pack(padx=3, pady=3)
        else:
            # 画像がない場合
            no_image_label = tk.Label(
                main_frame,
                text="[画像なし]",
                font=("Arial", 10),
                fg="#888888",
                bg="#2b2b2b"
            )
            no_image_label.pack(pady=(0, 15))

        # ナンバープレート情報表示（日本形式）
        plate_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief=tk.GROOVE)
        plate_frame.pack(fill=tk.X, pady=(0, 15))
        
        # ナンバープレート風の表示
        label_plate = tk.Label(
            plate_frame,
            text=self.plate_text,
            font=("MS Gothic", 18, "bold"),
            fg="#000000",
            bg="#ffffff",
            justify=tk.CENTER,
            padx=20,
            pady=15
        )
        label_plate.pack()

        # 閉じるボタン
        btn = tk.Button(
            main_frame,
            text="閉じる",
            font=("Arial", 11),
            width=12,
            command=root.destroy,
            bg="#4a4a4a",
            fg="#ffffff",
            activebackground="#666666",
            activeforeground="#ffffff",
            relief=tk.FLAT
        )
        btn.pack(pady=(5, 0))

        # Escキーで閉じる
        root.bind("<Escape>", lambda e: root.destroy())
        
        root.mainloop()
