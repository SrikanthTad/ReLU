from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor
import sklearn.datasets
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    batch_size = 256
    #X, y = sklearn.datasets.make_moons(3000, noise=0.1)
    X, y = sklearn.datasets.make_circles(1000, noise=0.03, factor = 0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    '''
    plt.figure()
    plt.title("Dataset")
    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Paired")
    plt.xlim([X[:, 0].min() - 1, X[:, 0].max() + 1])
    plt.ylim([X[:, 1].min() - 1, X[:, 1].max() + 1])
    plt.savefig('fig/dataset.png')
    '''

    X_train_t = torch.from_numpy(X_train).to(torch.float32)
    y_train_t = torch.from_numpy(y_train).to(torch.float32)
    X_test_t = torch.from_numpy(X_test).to(torch.float32)
    y_test_t = torch.from_numpy(y_test).to(torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    #train_dataset = mnist.MNIST(root='../LeNet-5/train', train=True, transform=ToTensor())
    #test_dataset = mnist.MNIST(root='../LeNet-5/test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    #model = Model()
    #sgd = SGD(model.parameters(), lr=1e-1)
    #cost = CrossEntropyLoss()
    #epoch = 100

    #torch.manual_seed(78)
    #torch.manual_seed(80)
    #torch.manual_seed(82)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    model = Model().to(device)
    #sgd = SGD(model.parameters(), lr=1e-2)
    sgd = torch.optim.Adam(model.parameters(), lr = 1.5e-2)
    #sgd = torch.optim.RMSprop(model.parameters(), lr = 1e-2)
    cost = CrossEntropyLoss()
    epoch = 100

    #np.set_printoptions(threshold=np.inf)

    '''
    print(model)

    ct = 0
    for name, param in model.named_parameters():
        ct += 1
        #param.requires_grad = False  #Freeze all layers
        print(str(ct) + ' ' + name) #Show the layers that contains parameters
        #if ct >= 25 and ct <= 44:
    if ct >= 5 and ct <= 1000:
        param.requires_grad = True  # Fine tune these layers
    else:
        param.requires_grad = False  # Freeze these layers
    '''

    res = []

    for _epoch in range(epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x, train_label = train_x.to(device), train_label.to(device)
            label_np = np.zeros((train_label.shape[0], 10))
            model.zero_grad() # model zero grad
            sgd.zero_grad()
            #print(train_x.float())
            predict_y = model(train_x.float())
            #print(predict_y.size())
            #print(train_label.size())
            loss = cost(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()
        
        '''
        if _epoch == 3:
            #torch.set_printoptions(profile="full")
            
            plt.figure()
            plt.hist(model.fc1.weights.grad.detach().cpu().numpy())
            plt.savefig('fig/weight1_grad.png')

            plt.figure()
            plt.hist(model.fc2.weights.grad.detach().cpu().numpy())
            plt.savefig('fig/weight2_grad.png')
        '''

        correct = 0
        _sum = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x, test_label = test_x.to(device), test_label.to(device)
            predict_y = model(test_x.float()).detach().cpu()
            predict_ys = np.argmax(predict_y, axis=-1)
            test_label1 = test_label.cpu()
            label_np = test_label1.numpy()
            _ = predict_ys == test_label1
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('accuracy: {:.2f}'.format(correct / _sum))
        torch.save(model.state_dict(), 'models/mnist_{:.2f}.pkl'.format(correct / _sum))
        res.append(correct / _sum)

    print(res)

    '''Draw decision boundary'''
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))
    #print(xx)
    #print(yy)
    #print(xx.shape)
    #print(yy.shape)
    #print(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).to(device).float())
    #print(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).shape)
    Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).to(device).float())
    #print(Z.shape)
    Z = Z.detach().cpu()
    Z1 = np.zeros(Z.shape[0])
    for i in range(Z.shape[0]):
        if Z[i][0] < Z[i][1]:
            Z1[i] = 1;
    Z1 = Z1.reshape(xx.shape)

    plt.figure()
    plt.title("Result")
    plt.contourf(xx, yy, Z1, cmap='Paired', alpha=0.5)
    #plt.axis('off')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired')
    plt.savefig('fig/result.png')

    '''Draw Surface'''
    Z_1 = Z[:, 0].numpy().reshape(xx.shape)
    Z_2 = Z[:, 1].numpy().reshape(xx.shape)
    #print("Z_1:", Z_1)
    #print("Z_2:", Z_2)
    #print(Z_1.shape)
    #print(Z_2.shape)
    plt.figure()
    plt.title("Surface")
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, Z_1, rstride=1, cstride=1,
                color = 'c', alpha = 0.9, edgecolor='none')
    ax.plot_surface(xx, yy, Z_2, rstride=1, cstride=1,
                color = (1.00, 0.42, 0.04), alpha = 0.9, edgecolor='none')
    plt.savefig('fig/surface.png')

    plt.show()
