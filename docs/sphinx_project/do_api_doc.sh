#clean up:
make clean;

#make doku
##configurations
sphinx-apidoc -o _source ../../reeds

cp ../../examples/*ipynb ./Examples

python conf.py

##execute making docu
make html
#make latex

cp -r _build/html/*  ../
