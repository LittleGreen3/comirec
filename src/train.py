# coding:utf-8
import argparse
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

# ============================================================================
# 环境变量和警告屏蔽配置（必须在导入 TensorFlow 之前）
# ============================================================================

# 1. 屏蔽 TensorFlow 警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示 ERROR，屏蔽 WARNING 和 INFO
os.environ['TF_MLIR_ENABLE_GPU_KERNEL_GEN'] = '0'  # 禁用 MLIR GPU kernel 生成警告
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# 2. CUDA 相关环境变量设置
os.environ['TF_CUDNN_WORKSPACE_LIMIT_IN_MB'] = '4096'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 3. CUDA 库路径配置（支持 conda 环境）
# Windows 和 Linux 都支持
if 'CONDA_PREFIX' in os.environ:
    conda_lib = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    if os.path.exists(conda_lib):
        # 检查是否有 CUDA 库
        try:
            files = os.listdir(conda_lib)
            has_cuda = any(f.startswith('libcudart') for f in files) or any(
                f.startswith('cudart64_') for f in files)
            if has_cuda:
                # Windows 使用 PATH，Linux 使用 LD_LIBRARY_PATH
                if sys.platform == 'win32':
                    current_path = os.environ.get('PATH', '')
                    if conda_lib not in current_path:
                        if current_path:
                            os.environ['PATH'] = f'{conda_lib};{current_path}'
                        else:
                            os.environ['PATH'] = conda_lib
                    # Windows 也可以使用 os.add_dll_directory (Python 3.8+)
                    if sys.version_info >= (3, 8):
                        try:
                            os.add_dll_directory(conda_lib)
                        except (OSError, AttributeError):
                            pass
                else:
                    # Linux/macOS
                    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
                    if conda_lib not in current_ld_path:
                        if current_ld_path:
                            os.environ['LD_LIBRARY_PATH'] = f'{conda_lib}:{current_ld_path}'
                        else:
                            os.environ['LD_LIBRARY_PATH'] = conda_lib
        except (OSError, PermissionError):
            pass

# 4. 导入 TensorFlow 并设置日志级别
import tensorflow as tf
# 进一步屏蔽 TensorFlow 的警告
tf.get_logger().setLevel('ERROR')

import faiss

from data_iterator import DataIterator
from model import (
    KerasModelDNN, KerasModelGRU4REC,
    KerasModelMIND, KerasModelComiRecDR, KerasModelComiRecSA
)


class TF2SummaryWriter(object):
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._writer = tf.summary.create_file_writer(log_dir)

    def add_scalar(self, tag, value, step):
        with self._writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self._writer.flush()

    def close(self):
        try:
            self._writer.close()
        except Exception:
            pass


# Tee stdout/stderr to both console and a log file
class Tee(object):
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, 'a', encoding='utf-8')

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--dataset', type=str, default='book', help='book | taobao')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='DNN', help='DNN | GRU4REC | MIND | ComiRec-DR | ComiRec-SA')
parser.add_argument('--learning_rate', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
parser.add_argument('--neg_num', type=int, default=10, help='negative samples per positive sample')
parser.add_argument('--test_iter', type=int, default=None,
                    help='evaluation interval in iterations (default: 500 for taobao, 1000 for book)')
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)

best_metric = 0


def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask


def scan_max_item_id(data_files):
    """
    扫描所有数据文件找出最大的 item_id
    
    Args:
        data_files: 数据文件路径列表（可以是None，表示跳过）
    
    Returns:
        最大 item_id，如果所有文件都不存在或为空则返回 0
    """
    max_item_id = 0
    
    for data_file in data_files:
        if data_file is None or not os.path.exists(data_file):
            continue
        
        try:
            file_max = 0
            with open(data_file, 'r') as f:
                for line in f:
                    conts = line.strip().split(',')
                    if len(conts) >= 2:
                        item_id = int(conts[1])
                        if item_id > 0:
                            file_max = max(file_max, item_id)
            max_item_id = max(max_item_id, file_max)
        except Exception:
            pass
    
    return max_item_id


def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate


def compute_diversity(item_list, item_cate_map):
    # 过滤掉那些没有 category 信息的 item
    filtered_items = [item for item in item_list if item in item_cate_map]
    n = len(filtered_items)
    if n < 2:
        return 0.0  # 如果有效 item 少于 2 个，无法计算多样性
    diversity = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diversity += item_cate_map[filtered_items[i]] != item_cate_map[filtered_items[j]]
    diversity /= ((n - 1) * n / 2)
    return diversity


def evaluate_full(test_data, keras_model, item_cate_map, topN, embedding_dim, model_type='DNN', coef=None,
                  save=True):
    # 获取 item 向量
    item_embs = keras_model.get_item_embeddings().numpy()

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    try:
        gpu_index = faiss.GpuIndexFlatIP(res, embedding_dim, flat_config)
        gpu_index.add(item_embs)
    except Exception:
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_diversity = 0.0

    for src, tgt in test_data:
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
        batch_size = len(item_id)

        # 使用output_user方法获取用户向量（对于多兴趣模型会返回所有兴趣向量）
        user_embs = keras_model.output_user(
            tf.convert_to_tensor(hist_item, dtype=tf.int32),
            tf.convert_to_tensor(hist_mask, dtype=tf.float32)
        ).numpy()

        # 判断是否为多兴趣模型（返回3维向量）
        if len(user_embs.shape) == 3:
            # 多兴趣模型：对每个兴趣向量分别搜索，然后合并结果
            ni = user_embs.shape[1]  # num_interest
            user_embs_reshaped = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs_reshaped, topN)

            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []

                # 合并多个兴趣向量的搜索结果
                item_list = list(
                    zip(np.reshape(I[i * ni:(i + 1) * ni], -1),
                        np.reshape(D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)

                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        item_cor_list.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break

                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)

                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)
        else:
            # 单向量模型：直接搜索
            D, I = gpu_index.search(user_embs, topN)

            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)
                for no, iid in enumerate(I[i]):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)

        total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total

    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}


def get_model(dataset, model_type, item_count, maxlen):
    if model_type == 'DNN':
        return KerasModelDNN(item_count, args.embedding_dim, args.hidden_size, maxlen)
    elif model_type == 'GRU4REC':
        return KerasModelGRU4REC(item_count, args.embedding_dim, args.hidden_size, maxlen)
    elif model_type == 'MIND':
        return KerasModelMIND(
            item_count, args.embedding_dim, args.hidden_size, args.num_interest, maxlen,
            hard_readout=True, relu_layer=(args.dataset == 'book')
        )
    elif model_type == 'ComiRec-DR':
        return KerasModelComiRecDR(item_count, args.embedding_dim, args.hidden_size, args.num_interest, maxlen)
    elif model_type == 'ComiRec-SA':
        return KerasModelComiRecSA(item_count, args.embedding_dim, args.hidden_size, args.num_interest, maxlen)
    else:
        return None


def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):
    extr_name = input('Please input the experiment name: ')
    para_name = '_'.join([dataset, model_type, 'b' + str(batch_size), 'lr' + str(lr), 'd' + str(args.embedding_dim),
                          'len' + str(maxlen)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('runs/' + exp_name) and save:
        flag = input('The exp name already exists. Do you want to cover? (y/n)')
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('runs/' + exp_name)
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name


def train(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        test_iter=50,
        model_type='DNN',
        lr=0.001,
        max_iter=100,
        patience=20
):
    global best_metric
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)
    # Prepare log file and redirect stdout/stderr to also write into file
    log_dir = os.path.join('runs', exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, 'train.log')
    sys.stdout = Tee(sys.stdout, log_file_path)
    sys.stderr = Tee(sys.stderr, log_file_path)

    best_model_path = "best_model/" + exp_name + '/'

    writer = TF2SummaryWriter('runs/' + exp_name)

    item_cate_map = load_item_cate(cate_file)

    keras_model = get_model(dataset, model_type, item_count, maxlen)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    neg_num = args.neg_num

    # 为多兴趣模型自动调整 patience
    if model_type in ['ComiRec-DR', 'ComiRec-SA', 'MIND'] and patience == 50:
        patience = 100

    # 注意：DataIterator 现在使用全局 random（已在主程序中设置 seed）
    # 这样可以确保每次运行的数据采样顺序一致，减少前期 recall 的波动
    train_data = DataIterator(train_file, batch_size, maxlen, train_flag=0)
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)

    # Checkpoint for model saving/restoring
    ckpt = tf.train.Checkpoint(model=keras_model, optimizer=optimizer)
    ckpt_dir = os.path.join(best_model_path, 'keras_ckpt')
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    latest_ckpt = ckpt_manager.latest_checkpoint
    if latest_ckpt:
        print(f"恢复 checkpoint: {latest_ckpt}")
        ckpt.restore(latest_ckpt)

    # Training step function with @tf.function for efficiency
    @tf.function
    def train_one_step(dummy_mid, hist_item, hist_mask, item_id, num_sampled):
        with tf.GradientTape() as tape:
            user_vec, item_vec = keras_model([
                dummy_mid, hist_item, hist_mask
            ], training=True)
            weights = keras_model.get_item_embeddings()
            biases = keras_model.item_bias
            labels = tf.reshape(item_id, [-1, 1])
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=weights,
                    biases=biases,
                    labels=labels,
                    inputs=user_vec,
                    num_sampled=num_sampled,
                    num_classes=item_count
                )
            )
        grads = tape.gradient(loss, keras_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, keras_model.trainable_variables))
        return loss

    print('training begin (Keras)')
    sys.stdout.flush()
    start_time = time.time()
    iter = 0
    loss_sum = 0.0
    trials = 0
    
    try:
        for src, tgt in train_data:
            nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)
            batch_size = len(item_id)
            dummy_mid = tf.convert_to_tensor(np.zeros((batch_size,), dtype=np.int32))
            hist_item_tensor = tf.convert_to_tensor(hist_item, dtype=tf.int32)
            hist_mask_tensor = tf.convert_to_tensor(hist_mask, dtype=tf.float32)
            item_id_tensor = tf.convert_to_tensor(item_id, dtype=tf.int32)

            num_sampled = neg_num * batch_size
            loss = train_one_step(dummy_mid, hist_item_tensor, hist_mask_tensor, item_id_tensor, num_sampled)
            loss_sum += float(loss.numpy())
            iter += 1

            if iter % test_iter == 0:
                metrics = evaluate_full(valid_data, keras_model, item_cate_map, args.topN, args.embedding_dim,
                                        args.model_type)
                log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)
                if metrics != {}:
                    log_str += ', ' + ', '.join(['valid ' + k + ': %.6f' % v for k, v in metrics.items()])
                print(exp_name)
                print(log_str)

                writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                if metrics != {}:
                    for key, value in metrics.items():
                        writer.add_scalar('eval/' + key, value, iter)

                if 'recall' in metrics:
                    recall = metrics['recall']
                    if recall > best_metric:
                        best_metric = recall
                        if not os.path.exists(best_model_path):
                            os.makedirs(best_model_path)
                        ckpt_manager.save()
                        trials = 0
                        print(f"🎉 Recall 提升至 {recall:.6f}，模型已保存")
                    else:
                        trials += 1
                        if trials > patience:
                            break

                loss_sum = 0.0
                test_time = time.time()
                print("time interval: %.4f min" % ((test_time - start_time) / 60.0))
                sys.stdout.flush()

            if iter >= max_iter * 1000:
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Restore best model for final evaluation
    best_ckpt = ckpt_manager.latest_checkpoint
    if best_ckpt:
        print(f'Restoring best checkpoint for evaluation: {best_ckpt}')
        ckpt.restore(best_ckpt).expect_partial()

    # 重新创建 valid_data 用于最终评估（确保数据顺序一致）
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=1)
    metrics = evaluate_full(valid_data, keras_model, item_cate_map, args.topN, args.embedding_dim,
                            args.model_type, save=False)
    print(', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()]))

    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    metrics = evaluate_full(test_data, keras_model, item_cate_map, args.topN, args.embedding_dim,
                            args.model_type, save=False)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def test(
        test_file,
        cate_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    item_cate_map = load_item_cate(cate_file)

    keras_model = get_model(dataset, model_type, item_count, maxlen)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    ckpt = tf.train.Checkpoint(model=keras_model, optimizer=optimizer)
    ckpt_dir = os.path.join(best_model_path, 'keras_ckpt')
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    latest_ckpt = ckpt_manager.latest_checkpoint
    if latest_ckpt:
        print(f"Restoring from checkpoint: {latest_ckpt}")
        ckpt.restore(latest_ckpt)
    else:
        print(f"No checkpoint found at {best_model_path}")
        return

    test_data = DataIterator(test_file, batch_size, maxlen, train_flag=2)
    metrics = evaluate_full(test_data, keras_model, item_cate_map, args.topN, args.embedding_dim,
                            args.model_type, coef=args.coef, save=False)
    print(', '.join(['test ' + key + ': %.6f' % value for key, value in metrics.items()]))


def output(
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        model_type='DNN',
        lr=0.001
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=False)
    best_model_path = "best_model/" + exp_name + '/'

    keras_model = get_model(dataset, model_type, item_count, maxlen)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    ckpt = tf.train.Checkpoint(model=keras_model, optimizer=optimizer)
    ckpt_dir = os.path.join(best_model_path, 'keras_ckpt')
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)
    latest_ckpt = ckpt_manager.latest_checkpoint
    if latest_ckpt:
        print(f"Restoring from checkpoint: {latest_ckpt}")
        ckpt.restore(latest_ckpt)
    else:
        print(f"No checkpoint found at {best_model_path}")
        return

    item_embs = keras_model.get_item_embeddings().numpy()
    if not os.path.exists('output'):
        os.makedirs('output')
    np.save('output/' + exp_name + '_emb.npy', item_embs)
    print(f"Item embeddings saved to output/{exp_name}_emb.npy")


if __name__ == '__main__':
    args = parser.parse_args()
    SEED = args.random_seed

    # 配置GPU设备
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) == 0:
        print("错误：未检测到GPU设备！")
        sys.exit(1)
    
    try:
        # 为每个GPU设置内存增长策略
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        tf.config.set_soft_device_placement(True)
    except RuntimeError as e:
        print(f"GPU配置失败: {e}")
        sys.exit(1)

    # Set random seeds
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'taobao':
        path = './data/taobao_data/'
        item_count = 1708531
        batch_size = 256
        maxlen = 50
        default_test_iter = 500
    elif args.dataset == 'book':
        path = './data/book_data/'
        item_count = 367983
        batch_size = 128
        maxlen = 20
        default_test_iter = 1000

    # 如果命令行指定了 test_iter，使用命令行参数；否则使用默认值
    test_iter = args.test_iter if args.test_iter is not None else default_test_iter

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    # 扫描所有数据文件找出最大 item_id，动态调整 item_count
    print("=" * 80)
    print("🔍 正在扫描数据文件以确定 item_id 范围...")
    print("=" * 80)
    data_files = [train_file, valid_file, test_file]
    max_item_id = scan_max_item_id(data_files)
    
    if max_item_id > 0:
        original_item_count = item_count
        # item_count 需要至少是 max_item_id + 1（因为 item_id 范围是 [1, item_count)）
        item_count = max(item_count, max_item_id + 1)
        print("=" * 80)
        print(f"✅ 扫描完成：最大 item_id = {max_item_id}")
        if item_count > original_item_count:
            print(f"📈 自动调整 item_count: {original_item_count} → {item_count}")
        else:
            print(f"✓  item_count ({item_count}) 已满足需求（最大 item_id + 1 = {max_item_id + 1}）")
        print("=" * 80)
        print()
    else:
        print("⚠️  警告：未能从数据文件中找到有效的 item_id，使用默认 item_count")
        print()

    if args.p == 'train':
        train(train_file=train_file, valid_file=valid_file, test_file=test_file, cate_file=cate_file,
              item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen, test_iter=test_iter,
              model_type=args.model_type, lr=args.learning_rate, max_iter=args.max_iter, patience=args.patience)
    elif args.p == 'test':
        test(test_file=test_file, cate_file=cate_file, item_count=item_count, dataset=dataset, batch_size=batch_size,
             maxlen=maxlen, model_type=args.model_type, lr=args.learning_rate)
    elif args.p == 'output':
        output(item_count=item_count, dataset=dataset, batch_size=batch_size, maxlen=maxlen,
               model_type=args.model_type, lr=args.learning_rate)
    else:
        print('do nothing...')