#!/usr/bin/env fish

meson setup builddir --buildtype release; or exit
meson compile -C builddir; or exit
exec ./plotwrap.py

