# click_selector.py
# マウスクリック位置から車BBOXを特定する

def select_track_id(click_x, click_y, tracked_cars):
    """
    click_x, click_y: マウスクリック座標
    tracked_cars:
    [
      { "track_id": int, "bbox": (x1,y1,x2,y2) }
    ]

    return:
      track_id or None
    """

    for car in tracked_cars:
        x1, y1, x2, y2 = car["bbox"]
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            return car["track_id"]

    return None
