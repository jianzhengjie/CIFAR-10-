import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid

# ---------------------- 数据加载与预处理 ----------------------
def load_cifar_batch(filename):
    """ 加载CIFAR-10批次数据 """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = np.array(datadict[b'labels'])
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).reshape(10000, -1)
        return X.astype(np.float32), Y

def load_cifar10(data_dir):
    """ 加载完整CIFAR-10数据集 """
    X_train, y_train = [], []
    for i in range(1,6):
        X, y = load_cifar_batch(os.path.join(data_dir, f'data_batch_{i}'))
        X_train.append(X)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_cifar_batch(os.path.join(data_dir, 'test_batch'))
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, y_train, X_test, test_size=0.1):
    """ 数据预处理与分割 """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
    X_train, X_val, X_test = X_train/255.0, X_val/255.0, X_test/255.0
    return X_train, y_train, X_val, y_val, X_test

# ---------------------- 数据增强 ----------------------
def augment_batch(X_batch):
    """ 实现随机水平翻转和随机裁剪 """
    flip_mask = np.random.rand(X_batch.shape[0]) > 0.5
    flipped = X_batch[flip_mask].reshape(-1,32,32,3)[:,:,::-1,:].reshape(-1,3072)
    X_batch[flip_mask] = flipped
    padded = np.pad(X_batch.reshape(-1,32,32,3), [(0,0), (4,4), (4,4), (0,0)], mode='constant')
    crop_x = np.random.randint(0, 9, size=X_batch.shape[0])
    crop_y = np.random.randint(0, 9, size=X_batch.shape[0])
    cropped = [padded[i, x:x+32, y:y+32].reshape(3072) for i, (x,y) in enumerate(zip(crop_x, crop_y))]
    return np.array(cropped)

# ---------------------- 神经网络模型 ----------------------
class ThreeLayerNN:
    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', use_bn=True):
        """ 初始化网络参数 """
        self.params = {}
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * np.sqrt(2./input_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, output_dim) * np.sqrt(2./hidden_dim)
        self.params['b2'] = np.zeros(output_dim)
        self.activation = activation
        self.use_bn = use_bn
        
        if use_bn:
            self.bn_params = {
                'gamma': np.ones(hidden_dim),
                'beta': np.zeros(hidden_dim),
                'running_mean': np.zeros(hidden_dim),
                'running_var': np.ones(hidden_dim)
            }

    def _activate(self, x):
        """ 激活函数 """
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        return x

    def _activate_deriv(self, x):
        """ 激活函数导数 """
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        return np.ones_like(x)

    def batchnorm_forward(self, x, mode='train'):
        """ BatchNorm前向传播 """
        gamma = self.bn_params['gamma']
        beta = self.bn_params['beta']
        running_mean = self.bn_params['running_mean']
        running_var = self.bn_params['running_var']
        
        if mode == 'train':
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_hat = (x - mu) / np.sqrt(var + 1e-5)
            self.bn_params['running_mean'] = 0.9 * running_mean + 0.1 * mu
            self.bn_params['running_var'] = 0.9 * running_var + 0.1 * var
        else:
            mu = running_mean
            var = running_var
            x_hat = (x - mu) / np.sqrt(var + 1e-5)
        out = gamma * x_hat + beta
        cache = (x, x_hat, mu, var, gamma)
        return out, cache

    def batchnorm_backward(self, dout, cache):
        """ BatchNorm反向传播 """
        x, x_hat, mu, var, gamma = cache
        N = x.shape[0]
        
        dgamma = np.sum(dout * x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)
        dx_hat = dout * gamma
        
        dvar = np.sum(dx_hat * (x - mu) * -0.5 * (var + 1e-5)**(-1.5), axis=0)
        dmu = np.sum(dx_hat * -1 / np.sqrt(var + 1e-5), axis=0) + dvar * np.mean(-2*(x - mu), axis=0)
        dx = dx_hat / np.sqrt(var + 1e-5) + dvar * 2*(x - mu)/N + dmu/N
        
        return dx, dgamma, dbeta

    def forward(self, X, mode='train'):
        """ 前向传播 """
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        h1_preact = np.dot(X, W1) + b1
        
        if self.use_bn:
            h1_bn, bn_cache = self.batchnorm_forward(h1_preact, mode)
            h1_act = self._activate(h1_bn)
        else:
            h1_act = self._activate(h1_preact)
            bn_cache = None
        
        scores = np.dot(h1_act, W2) + b2
        cache = (X, h1_preact, bn_cache, h1_act, scores)  # 将scores加入缓存
        return scores, cache

    def compute_loss(self, scores, y, reg):
        """ 计算交叉熵损失 """
        N = y.shape[0]
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        data_loss = -np.log(probs[np.arange(N), y]).mean()
        reg_loss = 0.5 * reg * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        return data_loss + reg_loss

    def backward(self, X, y, cache, reg):
        """ 反向传播 """
        X, h1_preact, bn_cache, h1_act, scores = cache  # 从缓存获取scores
        W1, W2 = self.params['W1'], self.params['W2']
        N = X.shape[0]
        
        # 输出层梯度计算
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N
        
        # 后续梯度计算保持不变...
        dW2 = np.dot(h1_act.T, dscores) + reg * W2
        db2 = np.sum(dscores, axis=0)
        dh1_act = np.dot(dscores, W2.T)
        
        # 隐藏层梯度
        dh1_preact = dh1_act * self._activate_deriv(h1_preact)
        
        if self.use_bn:
            dh1_bn, dgamma, dbeta = self.batchnorm_backward(dh1_preact, bn_cache)
            dW1 = np.dot(X.T, dh1_bn) + reg * W1
            db1 = np.sum(dh1_bn, axis=0)
        else:
            dW1 = np.dot(X.T, dh1_preact) + reg * W1
            db1 = np.sum(dh1_preact, axis=0)
        
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        if self.use_bn:
            grads['gamma'] = dgamma
            grads['beta'] = dbeta
            
        return grads

    def predict(self, X):
        """ 预测 """
        scores, _ = self.forward(X, mode='test')
        return np.argmax(scores, axis=1)

    def save_params(self, filename):
        """ 保存参数 """
        params = self.params.copy()
        if self.use_bn:
            params.update(self.bn_params)
        np.savez(filename, **params)

    def load_params(self, filename):
        """ 加载参数 """
        data = np.load(filename)
        for k in self.params:
            if k in data:
                self.params[k] = data[k]
        if self.use_bn:
            for k in self.bn_params:
                if k in data:
                    self.bn_params[k] = data[k]

# ---------------------- 优化器 ----------------------
class SGD:
    """ 普通SGD优化器 """
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def update(self, grads):
        for param in self.model.params:
            self.model.params[param] -= self.lr * grads[param]
        if self.model.use_bn:
            for p in ['gamma', 'beta']:
                self.model.bn_params[p] -= self.lr * grads[p]

class MomentumSGD:
    """ 带动量的SGD """
    def __init__(self, model, lr=0.01, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocities = {k: np.zeros_like(v) for k,v in model.params.items()}
        if model.use_bn:
            self.velocities['gamma'] = np.zeros_like(model.bn_params['gamma'])
            self.velocities['beta'] = np.zeros_like(model.bn_params['beta'])
        
    def update(self, grads):
        for param in self.model.params:
            self.velocities[param] = self.momentum * self.velocities[param] - self.lr * grads[param]
            self.model.params[param] += self.velocities[param]
        if self.model.use_bn:
            for param in ['gamma', 'beta']:
                self.velocities[param] = self.momentum * self.velocities[param] - self.lr * grads[param]
                self.model.bn_params[param] += self.velocities[param]

# ---------------------- 训练与评估 ----------------------
def train(model, optimizer, X_train, y_train, X_val, y_val, 
         epochs=50, batch_size=128, reg=1e-4, save_path=None, 
         lr_decay=0.95, verbose=True):
    """ 训练循环 """
    history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # 打乱数据
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        epoch_loss = 0.0
        for i in range(0, X_train.shape[0], batch_size):
            # 数据增强
            X_batch = augment_batch(X_train_shuffled[i:i+batch_size].copy())
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # 前向传播
            scores, cache = model.forward(X_batch, mode='train')
            loss = model.compute_loss(scores, y_batch, reg)
            
            # 反向传播
            grads = model.backward(X_batch, y_batch, cache, reg)
            
            # 参数更新
            optimizer.update(grads)
            
            epoch_loss += loss * X_batch.shape[0]
        
        # 学习率衰减
        if hasattr(optimizer, 'lr'):
            optimizer.lr *= lr_decay
        
        # 验证集评估
        val_acc, val_loss = evaluate(model, X_val, y_val, reg)
        history['train_loss'].append(epoch_loss / X_train.shape[0])
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc and save_path is not None:
            best_val_acc = val_acc
            model.save_params(save_path)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {epoch_loss/X_train.shape[0]:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    return history

def evaluate(model, X, y, reg=0.0):
    """ 模型评估 """
    scores, _ = model.forward(X, mode='test')
    loss = model.compute_loss(scores, y, reg)
    pred = np.argmax(scores, axis=1)
    acc = np.mean(pred == y)
    return acc, loss

# ---------------------- 可视化工具 ----------------------
def visualize_history(history, filename):
    """ 训练曲线可视化 """
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_weights(W, filename):
    """ 权重可视化 """
    W = W.reshape(32,32,3,-1)
    plt.figure(figsize=(12,12))
    for i in range(16):
        plt.subplot(4,4,i+1)
        wimg = 255 * (W[:,:,:,i] - W.min()) / (W.max() - W.min())
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
    plt.savefig(filename)
    plt.close()

# ---------------------- 主程序 ----------------------
if __name__ == '__main__':
    # 加载数据
    X_train, y_train, X_test, y_test = load_cifar10('./data/cifar-10-batches-py')
    X_train, y_train, X_val, y_val, X_test = preprocess_data(X_train, y_train, X_test)
    
    # 超参数搜索配置
    param_grid = {
        'hidden_dim': [256, 512],
        'lr': [0.01, 0.005],
        'reg': [1e-3, 1e-4],
        'activation': ['relu', 'sigmoid'],
        'optimizer': ['sgd', 'momentum']
    }
    
    results = []
    for i, params in enumerate(ParameterGrid(param_grid)):
        print(f"\n=== 实验 {i+1}/{len(ParameterGrid(param_grid))} ===")
        print("超参数:", params)
        
        # 初始化模型
        model = ThreeLayerNN(
            input_dim=3072,
            hidden_dim=params['hidden_dim'],
            output_dim=10,
            activation=params['activation']
        )
        
        # 选择优化器
        if params['optimizer'] == 'sgd':
            optimizer = SGD(model, lr=params['lr'])
        else:
            optimizer = MomentumSGD(model, lr=params['lr'])
        
        # 训练模型
        history = train(
            model, optimizer,
            X_train, y_train,
            X_val, y_val,
            epochs=75,
            batch_size=256,
            reg=params['reg'],
            save_path=f'best_model_{i}.npz',
            lr_decay=0.95
        )
        
        # 加载最佳模型并测试
        model.load_params(f'best_model_{i}.npz')
        test_acc, _ = evaluate(model, X_test, y_test)
        results.append({
            'params': params,
            'val_acc': max(history['val_acc']),
            'test_acc': test_acc
        })
        
        # 生成可视化
        visualize_history(history, f'training_history_{i}.png')
        visualize_weights(model.params['W1'], f'weights_visualization_{i}.png')
    
    # 输出最终结果
    print("\n=== 最终结果 ===")
# 初始化最佳结果跟踪变量
    best_test_acc = 0.0
    best_params = None
    
    for res in results:
        print(f"参数组合: {res['params']}")
        print(f"验证集最高准确率: {res['val_acc']:.4f}")
        print(f"测试集准确率: {res['test_acc']:.4f}\n")
        
        # 更新最佳测试准确率记录
        if res['test_acc'] > best_test_acc:
            best_test_acc = res['test_acc']
            best_params = res['params']
    
    # 单独打印最佳结果
    print("\n=== 最佳超参数组合 ===")
    print(f"测试准确率最高的参数组合: {best_params}")
    print(f"测试准确率: {best_test_acc:.4f}")
    print("对应的模型文件: best_model_{}.npz".format(
        results.index(next(r for r in results if r['test_acc'] == best_test_acc))
    ))