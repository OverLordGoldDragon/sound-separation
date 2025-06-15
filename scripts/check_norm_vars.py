import tensorflow.compat.v1 as tf

ckpt_vars = tf.train.list_variables('FUSS_baseline_model/baseline_model/baseline_model')
norm_vars = [name for name, shape in ckpt_vars if 'norm' in name.lower()]
print('Norm variables:')
for var in norm_vars[:20]:
    print(f'  {var}')

print('\nFirst few conv_block_0 variables:')
block0_vars = [name for name, shape in ckpt_vars if 'conv_block_0' in name]
for var in block0_vars[:20]:
    print(f'  {var}')