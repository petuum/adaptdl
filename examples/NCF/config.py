# dataset name 
import adaptdl.env
import os.path
dataset = 'pinterest-20'#'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20']

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = adaptdl.env.share_path()

train_rating = os.path.join(main_path, '{}.train.rating'.format(dataset))
test_rating = os.path.join(main_path, '{}.test.rating'.format(dataset))
test_negative = os.path.join(main_path, '{}.test.negative'.format(dataset))

model_path = os.path.join(main_path, 'models')
GMF_model_path = os.path.join(model_path, 'GMF.pth')
MLP_model_path = os.path.join(model_path, 'MLP.pth')
NeuMF_model_path = os.path.join(model_path, 'NeuMF.pth')
