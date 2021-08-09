infile=/home/linaiqi/Lab/data/hESC200.txt
outfile=/home/linaiqi/Lab/data/gene/tmp/gpgene_hESC200.txt
echo $outfile
python main.py --infile $infile --outfile $outfile --sizen 758 --sizem 20 --gene 200  --iter 40000 --ktype Poly2