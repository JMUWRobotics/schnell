# Run four cameras

to compile use:

```sh
meson setup builddir
meson compile -C builddir
```

then run the program with:

```sh
python3 viswrap.py
```

each argument needs to be passed with ```--sift``` as example

# example

Example for a flat water surface with 4 cameras:

```sh
python3 viswrap.py --solve --sift --lone 0.4 
```

# arguments

| Argument   | Description                     | Type   |
|------------|---------------------------------|--------|
| `--sin`    | Enables sine processing         | Flag   |
| `--lone`   | Sets the lone threshold value   | Float  |
| `--solve`  | Enable the ceres solver         | Flag   |