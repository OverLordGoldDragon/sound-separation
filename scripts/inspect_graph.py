import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

# Load the meta graph
tf.train.import_meta_graph('FUSS_baseline_model/baseline_model/baseline_inference.meta')
graph = tf.get_default_graph()

print("=== Collections ===")
for key in graph.get_all_collection_keys():
    collection = tf.get_collection(key)
    print(f"{key}: {len(collection)} items")
    if len(collection) < 10:
        for item in collection:
            print(f"  - {item}")

print("\n=== Placeholders ===")
placeholders = [op for op in graph.get_operations() if op.type == 'Placeholder']
for ph in placeholders:
    print(f"{ph.name}: {ph.outputs[0].shape}")

print("\n=== Operations with 'output' in name ===")
output_ops = [op for op in graph.get_operations() if 'output' in op.name.lower()]
for op in output_ops[:20]:
    print(f"{op.name}: {op.type}")

print("\n=== Operations with 'separated' in name ===")
sep_ops = [op for op in graph.get_operations() if 'separated' in op.name.lower()]
for op in sep_ops[:10]:
    print(f"{op.name}: {op.type}")

print("\n=== Last 20 operations ===")
all_ops = list(graph.get_operations())
for op in all_ops[-20:]:
    print(f"{op.name}: {op.type}")
    
print("\n=== Operations ending with numbers (likely outputs) ===")
num_ops = [op for op in graph.get_operations() if op.name.split('/')[-1].isdigit()]
for op in num_ops[-10:]:
    print(f"{op.name}: {op.type} -> {[out.shape for out in op.outputs]}")

print("\n=== Looking for final outputs ===")
final_ops = [op for op in graph.get_operations() if not any(out.consumers() for out in op.outputs)]
print(f"Found {len(final_ops)} operations with no consumers (potential outputs)")
for op in final_ops[:10]:
    print(f"{op.name}: {op.type}")