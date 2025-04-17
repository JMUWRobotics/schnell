#!/usr/bin/env fish

meson setup builddir --buildtype release; or exit
meson compile -C builddir; or exit
exec ./viswrap.py --lone=0.04 --solve --sift

