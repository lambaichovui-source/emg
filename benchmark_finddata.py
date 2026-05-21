import timeit
import sys
from PyQt6.QtWidgets import QApplication, QComboBox

app = QApplication(sys.argv)
combo = QComboBox()
TIMEBASE_OPTIONS = [5, 10, 20, 40]
for s in TIMEBASE_OPTIONS:
    combo.addItem(f"{s} sec", s)

timebase_map = {s: i for i, s in enumerate(TIMEBASE_OPTIONS)}

def use_finddata():
    combo.findData(20)
    combo.findData(5)

def use_dict():
    timebase_map.get(20, -1)
    timebase_map.get(5, -1)

if __name__ == '__main__':
    t1 = timeit.timeit(use_finddata, number=10000)
    t2 = timeit.timeit(use_dict, number=10000)
    print(f"findData: {t1:.6f} s")
    print(f"dict lookup: {t2:.6f} s")
