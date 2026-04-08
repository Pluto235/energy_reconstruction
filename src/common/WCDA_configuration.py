import os
import numpy as np
import pandas as pd

class WCDAConfig:

    def __init__(self, filename="/home/zhanghongfei/nn_rec/source/wcda_surveyx.txt"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} doesn't exist, please check file path.")

        self.filename = filename
        self.df = pd.read_csv(filename, delim_whitespace=True, comment="#")

        self.xy_offset = {
            0: (-75, -55),
            1: (75, -55),
            2: (0, 75)
        }

        self.cr_offset = {
            0: (0, 0),
            1: (30, 0),
            2: (0, 30)
        }

    def get_xy(self, idmc):
        idmc = np.atleast_1d(idmc)
        rows = self.df.iloc[idmc]
        s = rows["s"].to_numpy()
        x = rows["x[m]"].to_numpy()
        y = rows["y[m]"].to_numpy()

        for key, (dx, dy) in self.xy_offset.items():
            mask = (s == key)
            x[mask] += dx
            y[mask] += dy

        return x, y

    def get_col_row(self, idmc):
        idmc = np.atleast_1d(idmc)
        rows = self.df.iloc[idmc]
        s = rows["s"].to_numpy()
        col = rows["col"].to_numpy()
        row = rows["row"].to_numpy()

        for key, (dcol, drow) in self.cr_offset.items():
            mask = (s == key)
            col[mask] += dcol
            row[mask] += drow

        return col, row


# if __name__ == "__main__":
#     cfg = WCDAConfig("wcda_surveyx.txt")
#     print("Testing get_col_row(900):", cfg.get_col_row(900))
#     print("Testing get_col_row(1800):", cfg.get_col_row(1800))
#     print("Testing get_xy([900,1800]):", cfg.get_xy([900, 1800]))
