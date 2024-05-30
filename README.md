## Environment
Make sure you have installed:
- R 4.4.0
- Nim 2.0.4
- [rnim](https://github.com/SciNim/rnim.git)

## Running the code
Compile the *example.nim* file using 
```bash
nim c --app:lib example.nim
```
It will automatically generate a dynamic library *libexample.so* and an R file that loads it *example.R*