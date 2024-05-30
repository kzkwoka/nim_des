## Environment
Make sure you have installed:
- R 4.4.0
- Nim 2.0.4
- [rnim](https://github.com/SciNim/rnim.git)

### CEC Benchmark 2017
Download the code from [CEC2017](https://staff.elka.pw.edu.pl/~djagodzi/cec2017/cec2017_0.2.0.tar.gz).
Create package from tar code with:
```bash
R CMD INSTALL --build cec2017_0.2.0.tar.gz
```
Alternatively unpack the contents, compile using R compiler and install with devtools:
```bash
R CMD SHLIB cec2017/src/cec2017.c
```
```R
devtools::install('cec2017')
```
The package should now be ready to use in R.
```R
# Example
> library(cec2017)
> cec2017(1, c(2,2))
[1] 7964633169
```
## Running the code
Compile the *example.nim* file using 
```bash
nim c --app:lib example.nim
```
It will automatically generate a dynamic library *libexample.so* and an R file that loads it *example.R*