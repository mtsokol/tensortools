pip install .
pip install finch-tensor==0.2.14
pip install sparse==0.17.0

python examples/cpd_sparse.py 100 100 100 0.03
python examples/cpd_sparse.py 200 200 200 0.03
python examples/cpd_sparse.py 300 300 300 0.03
python examples/cpd_sparse.py 400 400 400 0.03

python examples/cpd_sparse.py 100 100 100 0.10
python examples/cpd_sparse.py 200 200 200 0.10
python examples/cpd_sparse.py 300 300 300 0.10

python examples/cpd_sparse.py 100 100 100 0.30
python examples/cpd_sparse.py 200 200 200 0.30
python examples/cpd_sparse.py 300 300 300 0.30
