[default]
all: postgpt pro

postgpt:
    gcc -O2 -o postgpt postgpt.c -lm

pro:
    # Windows does not run pro.exe without the increased stack
    gcc -O2 -o pro postgpt-pro.c -lm -Wl,--stack,8388608

clean:
    rm postgpt