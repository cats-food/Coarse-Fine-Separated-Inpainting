from main import main
main(mode=2)

# psv normal 630k_15k
'''
python test.py \
--model 4 \
--input "/home/ysy/batch_test/psv/__GT__" \
--mask "/home/ysy/batch_test/psv/__MASK__" \
--output "/home/ysy/batch_test/psv/FCA5.5_630k_15k" \
--G1 "/home/ysy/Projects/FCA 5.5/checkpoints/psv/CoarseModel_G_00630000.pth" \
--G2 "/home/ysy/Projects/FCA 5.5/checkpoints/psv/RefineModel_G_00015000.pth" 
'''

# psv noRes_630k_370k
'''
python test.py \
--model 4 \
--input "/home/ysy/batch_test/psv/__GT__" \
--mask "/home/ysy/batch_test/psv/__MASK__" \
--output "/home/ysy/batch_test/psv/FCA5.5(noRes)_630k_370k" \
--G1 "/home/ysy/Projects/FCA 5.5/checkpoints/psv/CoarseModel_G_00630000.pth" \
--G2 "/home/ysy/Projects/FCA 5.5/checkpoints/psv(noRes)/RefineModel_G_00370000.pth" 
'''
#psv noAtt_630k_30k
'''
python test.py \
--model 4 \
--input "/home/ysy/batch_test/psv/__GT__" \
--mask "/home/ysy/batch_test/psv/__MASK__" \
--output "/home/ysy/batch_test/psv/FCA5.5(noAtt)_630k_5k" \
--G1 "/home/ysy/Projects/FCA 5.5/checkpoints/psv/CoarseModel_G_00630000.pth" \
--G2 "/home/ysy/Projects/FCA 5.5/checkpoints/psv(noAtt)/RefineModel_G_00005000.pth" 
'''
#psv nonGateAtt_630k_20k
'''
python test.py \
--model 4 \
--input "/home/ysy/batch_test/psv/__GT__" \
--mask "/home/ysy/batch_test/psv/__MASK__" \
--output "/home/ysy/batch_test/psv/FCA5.5(nonGateAtt)_630k_5k" \
--G1 "/home/ysy/Projects/FCA 5.5/checkpoints/psv/CoarseModel_G_00630000.pth" \
--G2 "/home/ysy/Projects/FCA 5.5/checkpoints/psv(nonGateAtt)/RefineModel_G_00020000.pth" 
'''

# celeba normal
'''
python test.py \
--model 4 \
--input "/home/ysy/batch_test/celeba/__GT__" \
--mask "/home/ysy/batch_test/celeba/__MASK__" \
--output "/home/ysy/batch_test/celeba/FCA5.5_1160k_580k" \
--G1 "/home/ysy/Projects/FCA 5.5/checkpoints/celeba/CoarseModel_G_01160000.pth" \
--G2 "/home/ysy/Projects/FCA 5.5/checkpoints/celeba/RefineModel_G_00580000.pth" 
'''

