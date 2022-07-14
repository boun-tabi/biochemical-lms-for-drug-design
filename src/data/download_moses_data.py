import moses

train = moses.get_dataset('train')
test = moses.get_dataset('test')

with open('data/moses/train.txt', 'w') as f:
  f.write('\n'.join(train.tolist()))
with open('data/moses/test.txt', 'w') as f:
  f.write('\n'.join(test.tolist()))
