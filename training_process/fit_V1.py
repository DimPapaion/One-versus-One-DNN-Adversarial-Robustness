"""
Created on 23/10/2022

@author: Dimitrios Papaioannou
"""


from Utils import *
from tqdm import tqdm
from Plots.Plots import *
import matplotlib.pyplot as plt
import gc


class BaseClassifier(object):
    def __init__(self, Parameters):
        self.model = Parameters['model']
        self.model2 = Parameters['model2']
        self.optimizer = Parameters['optimizer']
        self.criterion = Parameters['criterion']
        self.scheduler = Parameters['scheduler']
        self.train_loader = Parameters['train_dl']
        self.valid_loader = Parameters['valid_dl']
        self.test_loader = Parameters['test_dl']
        self.num_classes = Parameters['num_classes']
        self.epochs = Parameters['epochs']
        self.device = Parameters['device']
        self.model_name = Parameters['model_name']
        self.path = Parameters['path']
        self.transformOVO = Parameters['matrix']

    def reverse_OvO_labels(self, y, transform):
        reverse_trans = transform.T
        return [torch.sum(torch.from_numpy(reverse_trans) * y[i], axis=1) for i in range(len(y))]

    def accuracy(self, y_true, y_scores):
        labels, preds = 0,0
        y_true = self.reverse_OvO_labels(y=y_true, transform=self.transformOVO)
        y_scores = self.reverse_OvO_labels(y=y_scores, transform=self.transformOVO)
        y_true = torch.tensor(np.array([torch.Tensor.cpu(y_true[i]).detach().numpy() for i in range(len(y_true))]))
        y_scores = torch.tensor(np.array([torch.Tensor.cpu(y_scores[i]).detach().numpy() for i in range(len(y_scores))]))
        _, labels = torch.max(y_true, dim=1)


        _, preds = torch.max(y_scores, dim=1)


        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def plot_losses(self, tr_loss, val_loss):
        """ Plot the losses in each epoch"""

        plt.plot(tr_loss, '-bx')
        plt.plot(val_loss, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.show()

    def training(self ):
        losst = 0.0
        acc = 0.0
        total = 0

        self.model = self.model.to(self.device)
        self.model.train(True)

        for data, target in tqdm(self.train_loader):

            # target = target.float()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            _,_,_,output = self.model(data)

            target = (target + 1) / 2
            output = (output + 1) / 2

            # loss = self.criterion(output, output3, output4, target2.argmax(dim=1))
            loss = self.criterion(output, target)

            losst += loss.item()  # * data.size(0)
            total += target.size(0)
            # AP += self.AP_scores(torch.Tensor.cpu(target.long()).detach().numpy(), torch.Tensor.cpu(output).detach().numpy())
            acc += self.accuracy(torch.Tensor.cpu(target).detach(), torch.Tensor.cpu(output).detach())

            loss.backward()
            self.optimizer.step()
            # if self.scheduler is not None:
            #     self.scheduler.step()


            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

        num_batches = float(len(self.train_loader))
        total_loss_ = losst / num_batches
        total_acc_ = acc / num_batches
        return total_loss_, total_acc_

    def evaluation(self):

        self.model.eval()
        if self.model2 is not None:
            self.model2.eval()
        losst = 0.0
        acc = 0.0
        total = 0.0

        with torch.no_grad():
            for data, target in tqdm(self.valid_loader):
                # target = target.float()
                data, target = data.to(self.device), target.to(self.device)
                _,_,_,output = self.model(data)

                target = (target + 1) / 2
                output = (output + 1) / 2

                loss = self.criterion(output, target)


                losst += loss.item()
                total += target.size(0)

                acc += self.accuracy(torch.Tensor.cpu(target).detach(), torch.Tensor.cpu(output).detach())

            num_batches = float(len(self.valid_loader))
            val_loss_ = losst / num_batches
            val_acc_ = acc / num_batches
        return val_loss_, val_acc_

    def train_model(self, validation=False):
        tr_loss, tr_acc = [], []
        val_loss, val_acc = [], []

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-------Epoch {}/{}---------".format(epoch + 1, self.epochs))

            if validation:
                tr_loss_, tr_acc_ = self.training()
                val_loss_, val_acc_ = self.evaluation()
                if self.scheduler is not None:
                    self.scheduler.step()

                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                val_loss.append(val_loss_), val_acc.append(val_acc_)

                print(
                    "Train Loss: {:.10f}, Train Accuracy: {:.6f}%, Validation Loss: {:.10f}, Validation Accuracy: {:.6f}%".format(
                        tr_loss_,
                        tr_acc_ * 100,
                        val_loss_,
                        val_acc_ * 100
                        ))

                # if epoch % 50 == 0:
                #     self.plot_losses(tr_loss, val_loss)


                if best_acc < val_acc_:

                    best_acc = val_acc_
                    checkpoint = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                  'optimizer_model': self.optimizer.state_dict()}
                    torch.save(checkpoint, './' + 'Cifar100' + '_OVO'  + '.tar')
                    print("-----Check Point in epoch {} saved in path: {} -----".format(int(epoch + 1),
                                                                                            str(self.path)))

            else:
                tr_loss_, tr_acc_ = self.training()

                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                print("Train Loss: {:.6f}, Train Accuracy: {:.4f}% ".format(tr_loss_,tr_acc_ * 100))
                if best_acc < tr_acc_:
                    best_acc = tr_acc_
                    # if epoch +1 == self.epochs:

                    torch.save(self.model.state_dict(), self.path)
                    print("-----Check Point in epoch {} saved in path: {} -----".format(int(epoch + 1), str(self.path)))

        if validation:
            return ([tr_loss, tr_acc], [val_loss, val_acc])
        else:
            return ([tr_loss, tr_acc])

    def test_model(self, model2=None):

        if model2 is None:
            self.model.eval()
        else:
            self.model = model2
            self.model.to(self.device)
            self.model.eval()
        self.model.to(self.device)
        losst = 0.0
        acc = 0.0

        y_pred = np.empty((0, self.num_classes), float)
        y_true = np.empty((0, self.num_classes), float)

        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                # target = target.float()
                data, target = data.to(self.device), target.to(self.device)

                bs, c, h, w = data.size()
                _,_,_,output = self.model(data.view(-1, c, h, w))
                loss = self.criterion(output, target)

                losst += loss
                acc += self.accuracy(torch.Tensor.cpu(target).detach().numpy(),
                                     torch.Tensor.cpu(output).detach().numpy())

                y_pred = np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)
                y_true = np.append(y_true, torch.Tensor.cpu(target).detach().numpy(), axis=0)

        num_samples = float(len(self.test_loader))
        test_loss = losst / num_samples
        test_acc = acc / num_samples

        print("test_loss: {:.6f}, test_Accuracy: {:.4f}%".format(test_loss, test_acc * 100))
        return test_loss, test_acc, y_pred, y_true


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MCSVDD_Training_(object):
    def __init__(self, Parameters):
        self.model = Parameters['model']
        self.model2 = Parameters['model2']
        self.model2 = Parameters['model3']
        self.optimizer = Parameters['optimizer']
        self.optimizer2 = Parameters['optimizer2']
        self.optimizer3 = Parameters['optimizer3']
        self.criterion = Parameters['criterion']
        self.criterion2 = Parameters['criterion2']
        self.criterion3 = Parameters['criterion3']
        self.scheduler = Parameters['scheduler']
        self.train_loader = Parameters['train_dl']
        self.valid_loader = Parameters['valid_dl']
        self.test_loader = Parameters['test_dl']
        self.num_classes = Parameters['num_classes']
        self.epochs = Parameters['epochs']
        self.device = Parameters['device']
        self.model_name = Parameters['model_name']
        self.path = Parameters['path']
        self.transformOVO = Parameters['matrix']

    def reverse_OvO_labels(self, y, transform):
        reverse_trans = transform.T
        return [torch.sum(torch.from_numpy(reverse_trans) * y[i], axis=1) for i in range(len(y))]

    def get_labels(self, y_true):
        y_true = self.reverse_OvO_labels(y=y_true, transform=self.transformOVO)
        y_true = torch.tensor(np.array([torch.Tensor.cpu(y_true[i]).detach().numpy() for i in range(len(y_true))]))
        _, labels = torch.max(y_true, dim=1)
        return labels

    def accuracy(self, y_true, y_scores):
        labels, preds = 0,0
        y_true = self.reverse_OvO_labels(y=y_true, transform=self.transformOVO)
        y_scores = self.reverse_OvO_labels(y=y_scores, transform=self.transformOVO)
        y_true = torch.tensor(np.array([torch.Tensor.cpu(y_true[i]).detach().numpy() for i in range(len(y_true))]))
        y_scores = torch.tensor(np.array([torch.Tensor.cpu(y_scores[i]).detach().numpy() for i in range(len(y_scores))]))
        _, labels = torch.max(y_true, dim=1)


        _, preds = torch.max(y_scores, dim=1)


        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def plot_losses(self, tr_loss, val_loss):
        """ Plot the losses in each epoch"""

        plt.plot(tr_loss, '-bx')
        plt.plot(val_loss, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.show()

    def training(self ):
        losst = 0.0
        acc = 0.0
        total = 0
        losses_xent = AverageMeter()
        losses_svdd1 = AverageMeter()
        losses_svdd2 = AverageMeter()
        losses_overall = AverageMeter()
        self.model, self.model1, self.model2= self.model.to(self.device), self.model1.to(self.device), self.model2.to(self.device)
        self.model.train(True)
        self.model1.train(True)
        self.model2.train(True)


        for data, target in tqdm(self.train_loader):

            # target = target.float()
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            feats_128, feats_256, feats_1024, outputsLinear  = self.model(data)

            target = (target + 1) / 2
            output = (output + 1) / 2
            labels = self.get_labels(y_true=target)

            # loss = self.criterion(output, output3, output4, target2.argmax(dim=1))
            outputsSVDD1 = self.model1(feats_256)
            outputsSVDD2 = self.model2(feats_1024)

            loss_xent = self.criterion(outputsLinear, target)
            # SVDD1 loss
            loss_svdd1 = self.criterion2(outputsSVDD1[0], outputsSVDD1[1], outputsSVDD1[2], labels)

            # SVDD2 loss
            loss_svdd2 = self.criterion3(outputsSVDD2[0], outputsSVDD2[1], outputsSVDD2[2], labels)
            # if epoch>10:
            loss = loss_xent + loss_svdd1 + loss_svdd2

            losst += loss.item()  # * data.size(0)
            total += target.size(0)
            # AP += self.AP_scores(torch.Tensor.cpu(target.long()).detach().numpy(), torch.Tensor.cpu(output).detach().numpy())
            acc += self.accuracy(torch.Tensor.cpu(target).detach(), torch.Tensor.cpu(output).detach())

            self.optimizer.zero_grad()
            self.optimizer2.zero_grad()
            self.optimizer3.zero_grad()

            loss.backward()
            self.optimizer.step()
            self.optimizer2.step()
            self.optimizer3.step()

            losses_xent.update(loss_xent.item(), labels.size(0))  # AverageMeter() has this param
            losses_svdd1.update(loss_svdd1.item(), labels.size(0))
            losses_svdd2.update(loss_svdd2.item(), labels.size(0))
            losses_overall.update(loss.item(), labels.size(0))
            # if self.scheduler is not None:
            #     self.scheduler.step()


            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

        num_batches = float(len(self.train_loader))
        total_loss_ = losst / num_batches
        total_acc_ = acc / num_batches
        return total_loss_, total_acc_

    def evaluation(self):

        self.model.eval()
        if self.model2 is not None:
            self.model2.eval()
        losst = 0.0
        acc = 0.0
        total = 0.0

        with torch.no_grad():
            for data, target in tqdm(self.valid_loader):
                # target = target.float()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                target = (target + 1) / 2
                output = (output + 1) / 2

                loss = self.criterion(output, target)


                losst += loss.item()
                total += target.size(0)

                acc += self.accuracy(torch.Tensor.cpu(target).detach(), torch.Tensor.cpu(output).detach())

            num_batches = float(len(self.valid_loader))
            val_loss_ = losst / num_batches
            val_acc_ = acc / num_batches
        return val_loss_, val_acc_

    def train_model(self, validation=False):
        tr_loss, tr_acc = [], []
        val_loss, val_acc = [], []

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-------Epoch {}/{}---------".format(epoch + 1, self.epochs))

            if validation:
                tr_loss_, tr_acc_ = self.training()
                val_loss_, val_acc_ = self.evaluation()
                if self.scheduler is not None:
                    self.scheduler.step()

                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                val_loss.append(val_loss_), val_acc.append(val_acc_)

                print(
                    "Train Loss: {:.10f}, Train Accuracy: {:.6f}%, Validation Loss: {:.10f}, Validation Accuracy: {:.6f}%".format(
                        tr_loss_,
                        tr_acc_ * 100,
                        val_loss_,
                        val_acc_ * 100
                        ))

                # if epoch % 50 == 0:
                #     self.plot_losses(tr_loss, val_loss)


                if best_acc < val_acc_:

                    best_acc = val_acc_
                    torch.save(self.model.state_dict(), self.path)
                    print("-----Check Point in epoch {} saved in path: {} -----".format(int(epoch + 1),
                                                                                            str(self.path)))

            else:
                tr_loss_, tr_acc_ = self.training()

                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                print("Train Loss: {:.6f}, Train Accuracy: {:.4f}% ".format(tr_loss_,tr_acc_ * 100))
                if best_acc < tr_acc_:
                    best_acc = tr_acc_
                    # if epoch +1 == self.epochs:

                    torch.save(self.model.state_dict(), self.path)
                    print("-----Check Point in epoch {} saved in path: {} -----".format(int(epoch + 1), str(self.path)))

        if validation:
            return ([tr_loss, tr_acc], [val_loss, val_acc])
        else:
            return ([tr_loss, tr_acc])

    def test_model(self, model2=None):

        if model2 is None:
            self.model.eval()
        else:
            self.model = model2
            self.model.to(self.device)
            self.model.eval()
        self.model.to(self.device)
        losst = 0.0
        acc = 0.0

        y_pred = np.empty((0, self.num_classes), float)
        y_true = np.empty((0, self.num_classes), float)

        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                # target = target.float()
                data, target = data.to(self.device), target.to(self.device)

                bs, c, h, w = data.size()
                output = self.model(data.view(-1, c, h, w))
                loss = self.criterion(output, target)

                losst += loss
                acc += self.accuracy(torch.Tensor.cpu(target).detach().numpy(),
                                     torch.Tensor.cpu(output).detach().numpy())

                y_pred = np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)
                y_true = np.append(y_true, torch.Tensor.cpu(target).detach().numpy(), axis=0)

        num_samples = float(len(self.test_loader))
        test_loss = losst / num_samples
        test_acc = acc / num_samples

        print("test_loss: {:.6f}, test_Accuracy: {:.4f}%".format(test_loss, test_acc * 100))
        return test_loss, test_acc, y_pred, y_true