import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.train.import_meta_graph('FUSS_baseline_model/baseline_model/baseline_inference.meta')
graph = tf.get_default_graph()

# Look for operations that might be the final output
candidates = []
for op in graph.get_operations():
    if ('inverse_stft' in op.name and 
        op.type not in ['Const', 'Pack', 'Assign'] and
        not any(out.consumers() for out in op.outputs)):
        candidates.append(op)

print("ISTFT output candidates:")
for op in candidates:
    print(f"  {op.name}: {op.type} -> {[out.shape for out in op.outputs]}")

# Also look for any operation with "separated" or "output" in the name
print("\nOther output candidates:")
for op in graph.get_operations():
    name_lower = op.name.lower()
    if (('separated' in name_lower or 'output' in name_lower) and 
        op.type not in ['Const', 'Pack', 'Assign'] and
        not any(out.consumers() for out in op.outputs)):
        print(f"  {op.name}: {op.type} -> {[out.shape for out in op.outputs]}")

# Look for the last non-assign operation
print("\nLast non-assign operations:")
non_assign_ops = [op for op in graph.get_operations() if op.type not in ['Assign', 'Const', 'Pack']]
for op in non_assign_ops[-10:]:
    print(f"  {op.name}: {op.type}")
    if not any(out.consumers() for out in op.outputs):
        print(f"    ^ This has no consumers (potential output)")