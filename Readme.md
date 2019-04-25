Rank 0
nohup python3 dist_pytorch_mnist.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 3 --root data0 &
Rank 1
nohup python3 dist_pytorch_mnist.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 3 --root data1 &
Rank 2
python3 dist_pytorch_mnist.py --init-method tcp://127.0.0.1:23456 --rank 2 --world-size 3 --root data2
