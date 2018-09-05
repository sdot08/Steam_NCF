import os

NUM_CHUNK = 4
OUTTER_EPOCH = 10

for i in range(OUTTER_EPOCH):
    for chunk_id in range(NUM_CHUNK):
        #THEANO_FLAGS=device=gpu,floatX=float32 python GMF.py --chunk_id 0 --dataset ml-1m --epochs 1 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
        args_list = 'chunk_id=' + str(chunk_id) + \
                    ' --dataset ml-1m \
                      --epochs 1 \
                      --batch_size 256 \
                      --num_factors 8 \
                      --regs [0,0] \
                      --num_neg 4 \
                      --lr 0.001 \
                      --learner adam \
                      --verbose 1 \
                      --out 1'
        os.system('THEANO_FLAGS=device=gpu,floatX=float32 python GMF.py ' + args_list)
