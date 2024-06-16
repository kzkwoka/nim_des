sudo apt update
sudo apt install r-base
sudo apt install nim

# Setup nim connection to R
git clone https://github.com/SciNim/rnim.git

# Setup CEC benchmark for R
wget https://staff.elka.pw.edu.pl/~djagodzi/cec2017/cec2017_0.2.0.tar.gz
R CMD INSTALL --build cec2017_0.2.0.tar.gz

# Setup CEC benchmark for nim
git clone https://github.com/ewarchul/cec.git
mv cec2017nim.c cec/

# Compile the nim implementation
nim --passC:"-Icec/include" --path:cec/src --app:lib c example.nim