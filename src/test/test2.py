import tensorflow as tf

a=tf.test.is_built_with_cuda()

b=tf.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)

print(a)
print(b)
