# midterm-1
训练与测试方法：
1.在下载测试集后指定测试集所在的文件夹地址
2.设置批处理大小
修改批处理大小以调整每次训练或测试时处理的数据量：BATCH_SIZE = 16
3.图像预处理
调整图像预处理步骤以适应你的需求，例如改变图像大小或归一化参数：
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
4.加载数据
使用下面的函数加载数据集，并获取数据加载器
train_class, train_loader = load_data(train_save_path, BATCH_SIZE)
test_class, test_loader = load_data(test_save_path, BATCH_SIZE)
5.定义和训练模型
可以通过修改以下部分来调整模型结构、损失函数和优化器：
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 200)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
6.训练模型
定义训练模型的函数，调整训练参数（如epoch数）：
def train_model(model, train_loader, criterion, optimizer, num_epochs=25, device='cpu', writer=None):
    # Training loop implementation
7.测试模型
定义测试模型的函数，用于评估模型的性能：
def test_model(model, test_loader, criterion, device='cpu', writer=None, epoch=None):
    # Testing loop implementation
8.进行实验
可以通过调整num_epochs_list和lr_list来尝试不同的训练epoch和学习率组合，从而找到最佳参数：
def experiment(num_epochs_list, lr_list, device='cpu'):
    # Experiment implementation
9.主函数
在主函数中运行实验，并记录和输出结果：
if __name__ == '__main__':
    num_epochs_list = [10, 15, 20]  # 可以修改epoch数
    lr_list = [0.001, 0.01, 0.1]    # 可以修改学习率

    results = experiment(num_epochs_list, lr_list, device='cpu')

    for num_epochs, lr, test_loss, test_acc in results:
        print(f"Epochs: {num_epochs}, LR: {lr}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
10.保存模型
在需要时，保存训练好的模型权重：
save_dir = r'C:\Users\23969\Desktop\CUB_200_2011\model_weights'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Add code to save model weights here
